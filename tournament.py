from typing import Callable

import numpy as np

from connect4.connect_state import ConnectState
from connect4.dtos import Game, Match, Participant, Versus


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def make_initial_matches(
    players: list[Participant], shuffle: bool, seed: int
) -> Versus:
    """Create the first round, padding with BYEs (None) up to a power of two."""
    players = players[:]  # copy
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(players)
    size = next_power_of_two(len(players))
    players += [None] * (size - len(players))  # BYEs
    return [(players[i], players[i + 1]) for i in range(0, len(players), 2)]


def play_round(
    versus: Versus,
    play_fn: Callable[[Participant, Participant, int, float, int], Participant],
    best_of: int,
    first_player_distribution: float,
    seed: int,
) -> list[Participant]:
    """Run a round and return the list of winners (handles BYEs)."""
    winners: list[Participant] = []
    for a, b in versus:
        if a is None and b is None:
            raise ValueError("Invalid match: two BYEs")
        if a is None:  # b advances
            winners.append(b)
        elif b is None:  # a advances
            winners.append(a)
        else:
            winners.append(play_fn(a, b, best_of, first_player_distribution, seed))
    return winners


def pair_next_round(winners: list[Participant]) -> Versus:
    """Pair adjacent winners for the next round."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def play(
    a: Participant,
    b: Participant,
    best_of: int,
    first_player_distribution: float,
    seed: int = 911,
) -> Participant:
    """Play a match between two participants and return the winner."""
    # Variables
    a_name, a_policy = a
    b_name, b_policy = b
    a_wins = 0
    b_wins = 0
    draws = 0
    total_games = 0
    games_to_win = (best_of // 2) + 1

    # Random Generator
    rng = np.random.default_rng(seed)

    games: list[Game] = []

    while a_wins < games_to_win and b_wins < games_to_win:
        total_games += 1
        # Decide who goes first based on the distribution.
        # Track which participant plays as which player number this game.
        if rng.random() < first_player_distribution:
            # a goes first (player -1), b goes second (player 1)
            first, second = (a, a_policy()), (b, b_policy())
            first_participant, second_participant = a, b
        else:
            # b goes first (player -1), a goes second (player 1)
            first, second = (b, b_policy()), (a, a_policy())
            first_participant, second_participant = b, a

        # Mount agents
        first[1].mount()
        second[1].mount()

        state = ConnectState()
        game_history: Game = Game()

        while not state.is_final():
            _, current_policy = first if state.player == -1 else second
            action = current_policy.act(state.board)
            game_history.append((state.board.copy().tolist(), int(action)))
            state = state.transition(int(action))

        games.append(game_history)

        # Attribute the win to the correct participant based on who played
        # which side this game — player -1 is first_participant, player 1 is second_participant.
        winner_value = state.get_winner()
        if winner_value == -1:
            if first_participant is a:
                a_wins += 1
            else:
                b_wins += 1
        elif winner_value == 1:
            if second_participant is a:
                a_wins += 1
            else:
                b_wins += 1
        else:
            draws += 1

        # Early stopping in case of too many draws
        if draws >= games_to_win + 5:
            break

    # Save match result
    match = Match(
        player_a=a_name,
        player_b=b_name,
        player_a_wins=a_wins,
        player_b_wins=b_wins,
        draws=draws,
        games=games,
    )

    # Save to file
    match_filename = f"match_{a_name}_vs_{b_name}.json"
    with open("versus/" + match_filename, "w") as f:
        f.write(match.model_dump_json(indent=4))

    if a_wins > 0 or b_wins > 0:
        return a if a_wins > b_wins else b
    # Decide winner at random in case of too many draws with no wins or tie
    return a if rng.random() < 0.5 else b


def run_tournament(
    players: list[Participant],
    play_fn: Callable[[Participant, Participant, int, float, int], Participant] = play,
    best_of: int = 7,
    first_player_distribution: float = 0.5,
    shuffle: bool = True,
    seed: int = 911,
):
    """
    Run a single-elimination tournament among the given players.

    Parameters
    ----------
    players : List[Participant]
        List of participants (name, policy) tuples.
    play_fn : Callable, optional
        Function that takes (a, b, best_of, first_player_distribution, seed)
        and returns the winner. Defaults to the module-level play().
    best_of : int, optional
        Number of games per match (default is 7).
    first_player_distribution : float, optional
        Probability that participant A goes first each game (default is 0.5).
    shuffle : bool, optional
        Whether to shuffle initial pairings (default is True).
    seed : int, optional
        Random seed for reproducibility (default is 911).
    """
    n_players = len(players)
    n_rounds = (next_power_of_two(n_players) - 1).bit_length()
    names = [p[0] for p in players]

    print("=" * 60)
    print(
        f"  TOURNAMENT  |  {n_players} players  |  best-of-{best_of}  |  {n_rounds} rounds"
    )
    print("=" * 60)
    print(f"  Participants: {', '.join(names)}")
    print(f"  Seed: {seed}  |  First-player bias: {first_player_distribution:.0%}")
    print("=" * 60)

    versus = make_initial_matches(players, shuffle=shuffle, seed=seed)

    round_num = 0
    while True:
        round_num += 1
        byes = [(a, b) for a, b in versus if a is None or b is None]
        real = [(a, b) for a, b in versus if a is not None and b is not None]

        print(f"\n  Round {round_num}")
        print(f"  {'─' * 40}")
        for a, b in real:
            print(f"    {a[0]:>20}  vs  {b[0]}")
        for a, b in byes:
            bye_name = (b or a)[0]
            print(f"    {bye_name:>20}  —  BYE (advances automatically)")

        winners = play_round(versus, play_fn, best_of, first_player_distribution, seed)

        print(f"\n  Results")
        print(f"  {'─' * 40}")
        for (a, b), winner in zip(versus, winners):
            if a is None or b is None:
                print(f"    {winner[0]:>20}  advances (bye)")
            else:
                loser = b if winner is a else a
                print(f"    {winner[0]:>20}  defeats  {loser[0]}")

        if len(winners) == 1:
            champion = winners[0]
            print(f"\n{'=' * 60}")
            print(f"  CHAMPION:  {champion[0]}")
            print(f"{'=' * 60}\n")
            return champion

        versus = pair_next_round(winners)
        next_names = [f"{a[0]} vs {b[0]}" for a, b in versus]
        print(f"\n  Next round: {' | '.join(next_names)}")
