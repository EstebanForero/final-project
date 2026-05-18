from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from .dataset import (
    HEIGHT,
    STRIDE,
    WIDTH,
    QValueRecord,
    QValuesDataset,
    decode_state_key,
    encode_board_current_player,
)
from .model import Connect4QNet
from .train import select_device, state_split_indices


@dataclass(frozen=True)
class StateActionTarget:
    action: int
    q_value: float
    visits: int


def legal_actions(state_key: int) -> set[int]:
    player_a_bits, player_b_bits, _ = decode_state_key(state_key)
    occupied = player_a_bits | player_b_bits
    actions: set[int] = set()

    for col in range(WIDTH):
        top_cell_bit = 1 << (col * STRIDE + HEIGHT - 1)
        if not occupied & top_cell_bit:
            actions.add(col)

    return actions


def load_model(path: Path, device: torch.device) -> Connect4QNet:
    model = Connect4QNet().to(device)
    model.load_state_dict(load_file(str(path)))
    model.eval()
    return model


def record_metrics(
    model: Connect4QNet,
    records: list[QValueRecord],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    weighted_sq_error = 0.0
    weighted_abs_error = 0.0
    weight_sum = 0.0
    sq_error = 0.0
    abs_error = 0.0
    count = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc="record eval", unit="batch"):
            batch = records[start : start + batch_size]
            xs = torch.stack(
                [encode_board_current_player(record.state_key) for record in batch]
            ).to(device)
            pred = model(xs).cpu()

            for row, record in enumerate(batch):
                error = float(pred[row, record.action].item()) - record.q_value
                weight = math.log1p(record.visits)

                sq_error += error * error
                abs_error += abs(error)
                weighted_sq_error += error * error * weight
                weighted_abs_error += abs(error) * weight
                weight_sum += weight
                count += 1

    return {
        "records": float(count),
        "mse": sq_error / max(count, 1),
        "rmse": math.sqrt(sq_error / max(count, 1)),
        "mae": abs_error / max(count, 1),
        "weighted_mse": weighted_sq_error / max(weight_sum, 1.0),
        "weighted_rmse": math.sqrt(weighted_sq_error / max(weight_sum, 1.0)),
        "weighted_mae": weighted_abs_error / max(weight_sum, 1.0),
    }


def grouped_state_metrics(
    model: Connect4QNet,
    records: list[QValueRecord],
    device: torch.device,
    batch_size: int,
    min_actions: int,
) -> dict[str, float]:
    grouped: dict[int, list[StateActionTarget]] = defaultdict(list)
    for record in records:
        grouped[record.state_key].append(
            StateActionTarget(record.action, record.q_value, record.visits)
        )

    state_items = [
        (state_key, targets)
        for state_key, targets in grouped.items()
        if len({target.action for target in targets}) >= min_actions
    ]

    top1_correct_visited = 0
    top1_correct_legal = 0
    total_gap_visited = 0.0
    total_gap_legal = 0.0
    evaluated = 0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(state_items), batch_size), desc="state eval", unit="batch"
        ):
            batch = state_items[start : start + batch_size]
            xs = torch.stack(
                [encode_board_current_player(state_key) for state_key, _ in batch]
            )
            pred = model(xs.to(device)).cpu()

            for row, (state_key, targets) in enumerate(batch):
                q_by_action = {target.action: target.q_value for target in targets}
                teacher_action = max(q_by_action, key=q_by_action.__getitem__)
                teacher_q = q_by_action[teacher_action]

                visited_actions = sorted(q_by_action)
                model_visited_action = max(
                    visited_actions, key=lambda action: float(pred[row, action].item())
                )
                top1_correct_visited += int(model_visited_action == teacher_action)
                total_gap_visited += teacher_q - q_by_action[model_visited_action]

                known_legal_actions = legal_actions(state_key)
                if known_legal_actions:
                    model_legal_action = max(
                        known_legal_actions,
                        key=lambda action: float(pred[row, action].item()),
                    )
                    top1_correct_legal += int(model_legal_action == teacher_action)
                    total_gap_legal += teacher_q - q_by_action.get(
                        model_legal_action, min(q_by_action.values())
                    )

                evaluated += 1

    return {
        "states": float(len(grouped)),
        "states_with_enough_actions": float(evaluated),
        "top1_accuracy_visited_actions": top1_correct_visited / max(evaluated, 1),
        "avg_q_gap_visited_actions": total_gap_visited / max(evaluated, 1),
        "top1_accuracy_legal_actions": top1_correct_legal / max(evaluated, 1),
        "avg_q_gap_legal_actions": total_gap_legal / max(evaluated, 1),
    }


def evaluate(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    print(f"device={device}")

    dataset = QValuesDataset(
        args.q_values,
        min_visits=args.min_visits,
        max_samples=args.max_samples,
    )
    print(f"samples={len(dataset)}")

    if args.split == "state-holdout":
        train_indices, eval_indices = state_split_indices(
            dataset,
            val_fraction=args.holdout_fraction,
            seed=args.seed,
        )
        print(
            f"split=state-holdout train_samples={len(train_indices)} "
            f"eval_samples={len(eval_indices)}"
        )
    else:
        eval_indices = list(range(len(dataset)))
        print("split=all")

    eval_records = [dataset.records[index] for index in eval_indices]
    model = load_model(Path(args.model), device)

    record_stats = record_metrics(
        model,
        eval_records,
        device,
        args.batch_size,
    )
    state_stats = grouped_state_metrics(
        model,
        eval_records,
        device,
        args.batch_size,
        args.min_actions,
    )

    print("\nrecord_metrics")
    for key, value in record_stats.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")

    print("\nstate_policy_metrics")
    for key, value in state_stats.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="connect4_qnet.safetensors")
    parser.add_argument("--q-values", default="q_values")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-visits", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-actions", type=int, default=2)
    parser.add_argument(
        "--split",
        choices=["state-holdout", "all"],
        default="state-holdout",
        help="Use state-holdout only with a model trained using --split-mode state.",
    )
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
