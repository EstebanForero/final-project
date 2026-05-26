from __future__ import annotations

import itertools
import importlib.util
import argparse
import inspect
import math
import multiprocessing as mp
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy

BENCHMARK_MODULE_PATH = Path(__file__).resolve()


@dataclass
class MoveTiming:
    agent: str
    seconds: float
    move_number: int
    selected_action: int | None
    had_win_opportunity: bool
    took_win: bool
    had_block_opportunity: bool
    blocked_opponent_win: bool


@dataclass
class GameResult:
    agent_a: str
    agent_b: str
    starter: str
    winner: str | None
    loser: str | None
    draw: bool
    invalid_move_by: str | None
    moves: int
    timings: list[MoveTiming]


def result_to_dict(result: GameResult) -> dict:
    return {
        "agent_a": result.agent_a,
        "agent_b": result.agent_b,
        "starter": result.starter,
        "winner": result.winner,
        "loser": result.loser,
        "draw": result.draw,
        "invalid_move_by": result.invalid_move_by,
        "moves": result.moves,
        "timings": [timing.__dict__ for timing in result.timings],
    }


def result_from_dict(data: dict) -> GameResult:
    return GameResult(
        agent_a=data["agent_a"],
        agent_b=data["agent_b"],
        starter=data["starter"],
        winner=data["winner"],
        loser=data["loser"],
        draw=data["draw"],
        invalid_move_by=data["invalid_move_by"],
        moves=data["moves"],
        timings=[MoveTiming(**timing) for timing in data["timings"]],
    )


def mount_agent(agent: Policy) -> None:
    try:
        agent.mount()
    except TypeError:
        agent.mount(None)


def instantiate_agent(agent_cls: type[Policy]) -> Policy:
    agent = agent_cls()
    mount_agent(agent)
    return agent


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def copy_agent_class(
    agent_name: str,
    agent_cls: type[Policy],
    game_index: int,
    side: str,
) -> tuple[type[Policy], Path]:
    """Load an agent class from a private per-game copy of its folder."""
    module = sys.modules[agent_cls.__module__]
    source_file = Path(module.__file__).resolve()
    source_dir = source_file.parent
    work_dir = Path(tempfile.mkdtemp(prefix=f"agent_game_{game_index}_{side}_"))
    copy_dir = work_dir / safe_name(agent_name)

    shutil.copytree(
        source_dir,
        copy_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__mojocache__"),
    )

    module_name = (
        f"_isolated_agent_{game_index}_{side}_"
        f"{safe_name(agent_name)}_{os.getpid()}_{time.time_ns()}"
    )

    # Some agents import side modules with fixed names, for example `solution`.
    # Remove them so the copied policy resolves files from its private folder.
    sys.modules.pop("solution", None)
    sys.path.insert(0, str(copy_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, copy_dir / "policy.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load policy module from {copy_dir / 'policy.py'}")
        copied_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = copied_module
        spec.loader.exec_module(copied_module)
    finally:
        try:
            sys.path.remove(str(copy_dir))
        except ValueError:
            pass

    copied_cls = getattr(copied_module, agent_cls.__name__)
    if not issubclass(copied_cls, Policy):
        raise TypeError(f"Copied class {copied_cls!r} is not a Policy")
    return copied_cls, work_dir


def class_to_spec(agent_cls: type[Policy]) -> dict[str, str]:
    module = sys.modules[agent_cls.__module__]
    source_file = Path(module.__file__).resolve()
    return {
        "source_dir": str(source_file.parent),
        "class_name": agent_cls.__name__,
    }


def load_agent_class_from_spec(
    agent_name: str,
    spec_data: dict[str, str],
    game_index: int,
    side: str,
    isolate_agent_files: bool,
) -> tuple[type[Policy], list[Path]]:
    source_dir = Path(spec_data["source_dir"]).resolve()
    work_dirs: list[Path] = []

    if isolate_agent_files:
        work_dir = Path(tempfile.mkdtemp(prefix=f"agent_game_{game_index}_{side}_"))
        policy_dir = work_dir / safe_name(agent_name)
        shutil.copytree(
            source_dir,
            policy_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__mojocache__"),
        )
        work_dirs.append(work_dir)
    else:
        policy_dir = source_dir

    module_name = (
        f"_subprocess_agent_{game_index}_{side}_"
        f"{safe_name(agent_name)}_{os.getpid()}_{time.time_ns()}"
    )

    sys.modules.pop("solution", None)
    sys.path.insert(0, str(policy_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, policy_dir / "policy.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load policy module from {policy_dir / 'policy.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(policy_dir))
        except ValueError:
            pass

    agent_cls = getattr(module, spec_data["class_name"], None)
    if agent_cls is None:
        candidates = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj is not Policy and issubclass(obj, Policy) and obj.__module__ == module.__name__
        ]
        if len(candidates) != 1:
            candidate_names = [candidate.__name__ for candidate in candidates]
            raise AttributeError(
                f"Could not find class {spec_data['class_name']!r} in {policy_dir / 'policy.py'} "
                f"and found Policy candidates {candidate_names!r}"
            )
        agent_cls = candidates[0]
    if not issubclass(agent_cls, Policy):
        raise TypeError(f"Loaded class {agent_cls!r} is not a Policy")
    return agent_cls, work_dirs


def play_fresh_game_from_specs(
    agent_a_name: str,
    agent_a_spec: dict[str, str],
    agent_b_name: str,
    agent_b_spec: dict[str, str],
    starter: str,
    game_seed: int | None,
    reseed_each_game: bool,
    isolate_agent_files: bool,
    game_index: int,
) -> GameResult:
    work_dirs: list[Path] = []
    agent_a_cls, work_dirs_a = load_agent_class_from_spec(
        agent_a_name, agent_a_spec, game_index, "a", isolate_agent_files
    )
    agent_b_cls, work_dirs_b = load_agent_class_from_spec(
        agent_b_name, agent_b_spec, game_index, "b", isolate_agent_files
    )
    work_dirs.extend(work_dirs_a)
    work_dirs.extend(work_dirs_b)
    try:
        return play_fresh_game(
            agent_a_name,
            agent_a_cls,
            agent_b_name,
            agent_b_cls,
            starter=starter,
            game_seed=game_seed,
            reseed_each_game=reseed_each_game,
            isolate_agent_files=False,
            game_index=game_index,
        )
    finally:
        for work_dir in work_dirs:
            shutil.rmtree(work_dir, ignore_errors=True)


def task_to_payload(task: tuple) -> dict:
    (
        game_index,
        agent_a_name,
        agent_a_cls,
        agent_b_name,
        agent_b_cls,
        starter,
        game_seed,
        reseed_each_game,
        isolate_agent_files,
    ) = task
    return {
        "game_index": game_index,
        "agent_a_name": agent_a_name,
        "agent_a_spec": class_to_spec(agent_a_cls),
        "agent_b_name": agent_b_name,
        "agent_b_spec": class_to_spec(agent_b_cls),
        "starter": starter,
        "game_seed": game_seed,
        "reseed_each_game": reseed_each_game,
        "isolate_agent_files": isolate_agent_files,
    }


def run_worker_payload(payload: dict) -> tuple[int, GameResult]:
    game_index = int(payload["game_index"])
    result = play_fresh_game_from_specs(
        payload["agent_a_name"],
        payload["agent_a_spec"],
        payload["agent_b_name"],
        payload["agent_b_spec"],
        starter=payload["starter"],
        game_seed=payload["game_seed"],
        reseed_each_game=payload["reseed_each_game"],
        isolate_agent_files=payload["isolate_agent_files"],
        game_index=game_index,
    )
    return game_index, result


def run_worker_cli(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    game_index, result = run_worker_payload(payload)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"game_index": game_index, "result": result_to_dict(result)}, f)


def immediate_winning_cols(board: np.ndarray, player: int) -> set[int]:
    state = ConnectState(board=board, player=player)
    winning_cols = set()
    for col in state.get_free_cols():
        try:
            if state.transition(col).get_winner() == player:
                winning_cols.add(col)
        except ValueError:
            pass
    return winning_cols


def play_fresh_game(
    agent_a_name: str,
    agent_a_cls: type[Policy],
    agent_b_name: str,
    agent_b_cls: type[Policy],
    starter: str,
    game_seed: int | None = None,
    reseed_each_game: bool = True,
    isolate_agent_files: bool = False,
    game_index: int = 0,
    max_moves: int = ConnectState.ROWS * ConnectState.COLS,
) -> GameResult:
    if reseed_each_game and game_seed is not None:
        random.seed(game_seed)
        np.random.seed(game_seed % (2**32 - 1))

    isolated_work_dirs: list[Path] = []
    if isolate_agent_files:
        agent_a_cls, work_dir_a = copy_agent_class(agent_a_name, agent_a_cls, game_index, "a")
        agent_b_cls, work_dir_b = copy_agent_class(agent_b_name, agent_b_cls, game_index, "b")
        isolated_work_dirs.extend([work_dir_a, work_dir_b])

    agent_a = instantiate_agent(agent_a_cls)
    agent_b = instantiate_agent(agent_b_cls)

    if starter == agent_a_name:
        player_to_name = {-1: agent_a_name, 1: agent_b_name}
        name_to_agent = {agent_a_name: agent_a, agent_b_name: agent_b}
    elif starter == agent_b_name:
        player_to_name = {-1: agent_b_name, 1: agent_a_name}
        name_to_agent = {agent_a_name: agent_a, agent_b_name: agent_b}
    else:
        raise ValueError(f"starter must be {agent_a_name!r} or {agent_b_name!r}")

    state = ConnectState()
    timings: list[MoveTiming] = []
    invalid_move_by = None
    loser = None
    winner = None

    for move_number in range(1, max_moves + 1):
        if state.is_final():
            break

        current_name = player_to_name[state.player]
        current_agent = name_to_agent[current_name]
        current_player = state.player
        win_cols = immediate_winning_cols(state.board, current_player)
        block_cols = immediate_winning_cols(state.board, -current_player)

        start = time.perf_counter()
        action = None
        try:
            action = int(current_agent.act(state.board.copy()))
        except Exception:
            invalid_move_by = current_name
        elapsed = time.perf_counter() - start

        timings.append(
            MoveTiming(
                agent=current_name,
                seconds=elapsed,
                move_number=move_number,
                selected_action=action,
                had_win_opportunity=bool(win_cols),
                took_win=action in win_cols if action is not None else False,
                had_block_opportunity=bool(block_cols),
                blocked_opponent_win=action in block_cols if action is not None else False,
            )
        )

        if invalid_move_by is not None or action is None or not state.is_applicable(action):
            invalid_move_by = current_name
            loser = current_name
            winner = agent_b_name if current_name == agent_a_name else agent_a_name
            break

        state = state.transition(action)

    if winner is None:
        winning_player = state.get_winner()
        if winning_player == 0:
            draw = True
        else:
            draw = False
            winner = player_to_name[winning_player]
            loser = agent_b_name if winner == agent_a_name else agent_a_name
    else:
        draw = False

    result = GameResult(
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        starter=starter,
        winner=winner,
        loser=loser,
        draw=draw,
        invalid_move_by=invalid_move_by,
        moves=len(timings),
        timings=timings,
    )
    for work_dir in isolated_work_dirs:
        shutil.rmtree(work_dir, ignore_errors=True)
    return result


AGENT_LOCKS = defaultdict(threading.Lock)


@contextmanager
def locked_agents(*agent_names: str):
    locks = [AGENT_LOCKS[name] for name in sorted(set(agent_names))]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def play_locked_game(*task):
    (
        game_index,
        agent_a_name,
        agent_a_cls,
        agent_b_name,
        agent_b_cls,
        starter,
        game_seed,
        reseed_each_game,
        isolate_agent_files,
    ) = task
    with locked_agents(agent_a_name, agent_b_name):
        result = play_fresh_game(
            agent_a_name,
            agent_a_cls,
            agent_b_name,
            agent_b_cls,
            starter=starter,
            game_seed=game_seed,
            reseed_each_game=reseed_each_game,
            isolate_agent_files=isolate_agent_files,
            game_index=game_index,
        )
    return game_index, result


def play_unlocked_game(*task):
    (
        game_index,
        agent_a_name,
        agent_a_cls,
        agent_b_name,
        agent_b_cls,
        starter,
        game_seed,
        reseed_each_game,
        isolate_agent_files,
    ) = task
    result = play_fresh_game(
        agent_a_name,
        agent_a_cls,
        agent_b_name,
        agent_b_cls,
        starter=starter,
        game_seed=game_seed,
        reseed_each_game=reseed_each_game,
        isolate_agent_files=isolate_agent_files,
        game_index=game_index,
    )
    return game_index, result


def make_non_overlapping_batches(schedule: list[tuple]) -> list[list[tuple]]:
    pending = list(schedule)
    batches = []
    while pending:
        used_agents = set()
        batch = []
        remaining = []
        for task in pending:
            _, name_a, _, name_b, _, _, _, _, _ = task
            if name_a not in used_agents and name_b not in used_agents:
                batch.append(task)
                used_agents.update([name_a, name_b])
            else:
                remaining.append(task)
        batches.append(batch)
        pending = remaining
    return batches


def run_process_round_robin(
    schedule: list[tuple],
    max_workers: int,
    allow_agent_overlap: bool,
    process_start_method: str,
    isolate_agent_files: bool,
) -> list[tuple[int, GameResult]]:
    indexed_results = []
    total_games = len(schedule)
    process_context = mp.get_context(process_start_method)

    if allow_agent_overlap:
        print(f"Running {total_games} games across {max_workers} worker processes")
        done_count = 0
        chunks = [
            schedule[start : start + max_workers]
            for start in range(0, len(schedule), max_workers)
        ]
        for chunk_index, chunk in enumerate(chunks, start=1):
            # With isolated files, fresh worker processes per chunk avoid native modules
            # such as Mojo binding the same type multiple times in a reused worker.
            chunk_workers = len(chunk) if isolate_agent_files else max_workers
            with ProcessPoolExecutor(max_workers=chunk_workers, mp_context=process_context) as executor:
                futures = [executor.submit(play_unlocked_game, *task) for task in chunk]
                for future in as_completed(futures):
                    game_index, result = future.result()
                    done_count += 1
                    indexed_results.append((game_index, result))
                    outcome = "draw" if result.draw else f"winner={result.winner}"
                    print(
                        f"[{done_count}/{total_games}] chunk={chunk_index} game={game_index} "
                        f"{result.agent_a} vs {result.agent_b}, starter={result.starter}, {outcome}"
                    )
        return indexed_results

    batches = make_non_overlapping_batches(schedule)
    print(
        f"Running {total_games} games across {max_workers} worker processes "
        f"in {len(batches)} non-overlapping batches"
    )
    done_count = 0
    for batch_index, batch in enumerate(batches, start=1):
        batch_workers = min(max_workers, len(batch))
        with ProcessPoolExecutor(max_workers=batch_workers, mp_context=process_context) as executor:
            futures = [executor.submit(play_unlocked_game, *task) for task in batch]
            for future in as_completed(futures):
                game_index, result = future.result()
                done_count += 1
                indexed_results.append((game_index, result))
                outcome = "draw" if result.draw else f"winner={result.winner}"
                print(
                    f"[{done_count}/{total_games}] batch={batch_index} game={game_index} "
                    f"{result.agent_a} vs {result.agent_b}, starter={result.starter}, {outcome}"
                )
    return indexed_results


def run_thread_round_robin(schedule: list[tuple], max_workers: int) -> list[tuple[int, GameResult]]:
    indexed_results = []
    total_games = len(schedule)
    print(f"Running {total_games} games with {max_workers} worker threads")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(play_locked_game, *task) for task in schedule]
        for done_count, future in enumerate(as_completed(futures), start=1):
            game_index, result = future.result()
            indexed_results.append((game_index, result))
            outcome = "draw" if result.draw else f"winner={result.winner}"
            print(
                f"[{done_count}/{total_games}] game={game_index} "
                f"{result.agent_a} vs {result.agent_b}, starter={result.starter}, {outcome}"
            )
    return indexed_results


def run_subprocess_task(task: tuple) -> tuple[int, GameResult]:
    payload = task_to_payload(task)
    with tempfile.TemporaryDirectory(prefix="agent_game_payload_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.json"
        output_path = temp_path / "output.json"
        with input_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_MODULE_PATH),
                "--worker",
                str(input_path),
                str(output_path),
            ],
            cwd=str(BENCHMARK_MODULE_PATH.parent),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Subprocess game failed\n"
                f"task={payload}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        # Preserve agent prints, such as Rollout_2 timing logs, in the notebook output.
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return int(data["game_index"]), result_from_dict(data["result"])


def run_subprocess_round_robin(schedule: list[tuple], max_workers: int) -> list[tuple[int, GameResult]]:
    indexed_results = []
    total_games = len(schedule)
    print(f"Running {total_games} games across {max_workers} fresh Python subprocesses")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_subprocess_task, task) for task in schedule]
        for done_count, future in enumerate(as_completed(futures), start=1):
            game_index, result = future.result()
            indexed_results.append((game_index, result))
            outcome = "draw" if result.draw else f"winner={result.winner}"
            print(
                f"[{done_count}/{total_games}] game={game_index} "
                f"{result.agent_a} vs {result.agent_b}, starter={result.starter}, {outcome}"
            )
    return indexed_results


def run_round_robin(
    agents: dict[str, type[Policy]],
    games_per_pair: int,
    max_workers: int,
    backend: str,
    seed: int,
    allow_agent_overlap: bool,
    process_start_method: str,
    reseed_each_game: bool,
    isolate_agent_files: bool,
) -> list[GameResult]:
    schedule = []
    game_index = 0
    for (name_a, cls_a), (name_b, cls_b) in itertools.combinations(agents.items(), 2):
        for game_no in range(games_per_pair):
            starter = name_a if game_no % 2 == 0 else name_b
            game_index += 1
            game_seed = seed + game_index * 10_007
            schedule.append(
                (
                    game_index,
                    name_a,
                    cls_a,
                    name_b,
                    cls_b,
                    starter,
                    game_seed,
                    reseed_each_game,
                    isolate_agent_files,
                )
            )

    total_games = len(schedule)
    max_workers = max(1, min(max_workers, total_games))
    if backend == "processes" and not allow_agent_overlap:
        max_batch_size = max((len(batch) for batch in make_non_overlapping_batches(schedule)), default=1)
        effective_parallelism = min(max_workers, max_batch_size)
    else:
        effective_parallelism = max_workers

    print("Parallel configuration")
    print(f"- backend: {backend}")
    print(f"- CPU cores detected: {os.cpu_count()}")
    print(f"- requested workers: {max_workers}")
    print(f"- effective concurrent games: {effective_parallelism}")
    print(f"- agent overlap allowed: {allow_agent_overlap}")
    print(f"- isolated agent files: {isolate_agent_files}")
    if backend == "processes":
        print(f"- process start method: {process_start_method}")
    if backend == "processes" and max_workers > 1:
        print("- execution mode: multi-core worker processes")
    elif backend == "subprocesses" and max_workers > 1:
        print("- execution mode: fresh Python subprocesses, multi-core and Mojo-safe")
    elif backend == "threads" and max_workers > 1:
        print("- execution mode: worker threads, limited by the Python GIL for CPU-bound code")
    else:
        print("- execution mode: serial")
    print()

    if backend == "processes" and max_workers > 1:
        indexed_results = run_process_round_robin(
            schedule, max_workers, allow_agent_overlap, process_start_method, isolate_agent_files
        )
    elif backend == "subprocesses" and max_workers > 1:
        indexed_results = run_subprocess_round_robin(schedule, max_workers)
    elif backend == "threads" and max_workers > 1:
        indexed_results = run_thread_round_robin(schedule, max_workers)
    else:
        print(f"Running {total_games} games serially")
        indexed_results = [play_unlocked_game(*task) for task in schedule]

    indexed_results.sort(key=lambda item: item[0])
    return [result for _, result in indexed_results]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("INPUT_JSON", "OUTPUT_JSON"))
    args = parser.parse_args()
    if args.worker:
        input_path, output_path = args.worker
        run_worker_cli(Path(input_path), Path(output_path))


if __name__ == "__main__":
    main()
