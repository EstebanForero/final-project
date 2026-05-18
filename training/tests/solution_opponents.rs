use std::{collections::HashMap, path::Path};

use rand::{RngExt, SeedableRng, rngs::StdRng};
use training::{
    connect4::Connect4Env,
    policy_improver::{GreedyPolicy, UcbPolicy},
    q_value_persistence::QValuePersistence,
    trial_gen::MonteCarloTreeSearch,
    types::{CurrentState, HEIGHT, Player, QValues, WIDTH},
};

const SEARCH_BUDGET: usize = 100;
const RANDOM_GAMES: usize = 10;

#[derive(Debug)]
struct GameReport {
    outcome: CurrentState,
    moves: usize,
}

#[derive(Clone, Copy, Debug)]
enum SolutionSide {
    Red,
    Yellow,
}

impl SolutionSide {
    fn player(self) -> Player {
        match self {
            SolutionSide::Red => Player::A,
            SolutionSide::Yellow => Player::B,
        }
    }

    fn label(self) -> &'static str {
        match self {
            SolutionSide::Red => "red",
            SolutionSide::Yellow => "yellow",
        }
    }
}

#[derive(Default)]
struct OutcomeStats {
    wins: usize,
    draws: usize,
    losses: usize,
}

impl OutcomeStats {
    fn record(&mut self, outcome: CurrentState, side: SolutionSide) {
        match outcome {
            CurrentState::Win(winner) if winner == side.player() => self.wins += 1,
            CurrentState::Win(_) => self.losses += 1,
            CurrentState::Draw => self.draws += 1,
            CurrentState::Ongoing => unreachable!("games must finish before being recorded"),
        }
    }
}

enum Opponent {
    Random(StdRng),
    FixedColumn(usize),
}

impl Opponent {
    fn choose_action(&mut self, valid_actions: &[usize]) -> usize {
        assert!(
            !valid_actions.is_empty(),
            "opponent cannot choose action: no valid actions available"
        );

        match self {
            Opponent::Random(rng) => {
                let index = rng.random_range(0..valid_actions.len());
                valid_actions[index]
            }
            Opponent::FixedColumn(column) => {
                if valid_actions.contains(column) {
                    *column
                } else {
                    valid_actions[0]
                }
            }
        }
    }
}

type Solution = MonteCarloTreeSearch<UcbPolicy, GreedyPolicy>;

fn load_q_values() -> QValues {
    if Path::new("q_values").exists() {
        QValuePersistence::new("q_values".into()).load_q_values()
    } else {
        QValues {
            table: HashMap::new(),
        }
    }
}

fn build_solution() -> Solution {
    MonteCarloTreeSearch::new(UcbPolicy::new(1.4), GreedyPolicy::new(), SEARCH_BUDGET)
}

fn play_solution_game(opponent: &mut Opponent, side: SolutionSide) -> GameReport {
    let mut env = Connect4Env::<WIDTH, HEIGHT>::new();
    let mut q_values = load_q_values();
    let mut solution = build_solution();

    for moves in 0..(WIDTH * HEIGHT) {
        if let Some(report) = finished_report(&env, moves) {
            return report;
        }

        play_turn(&mut env, &mut solution, &mut q_values, opponent, side);
    }

    assert_ne!(
        env.current_state,
        CurrentState::Ongoing,
        "game did not finish within the maximum number of moves"
    );

    GameReport {
        outcome: env.current_state,
        moves: WIDTH * HEIGHT,
    }
}

fn finished_report(env: &Connect4Env<WIDTH, HEIGHT>, moves: usize) -> Option<GameReport> {
    (env.current_state != CurrentState::Ongoing).then_some(GameReport {
        outcome: env.current_state,
        moves,
    })
}

fn play_turn(
    env: &mut Connect4Env<WIDTH, HEIGHT>,
    solution: &mut Solution,
    q_values: &mut QValues,
    opponent: &mut Opponent,
    side: SolutionSide,
) {
    let valid_actions = env.valid_actions();
    let action = choose_action(env, solution, q_values, opponent, &valid_actions, side);

    assert!(
        valid_actions.contains(&action),
        "selected invalid action {action}; valid actions were {valid_actions:?}"
    );

    env.play_move(action);
}

fn choose_action(
    env: &Connect4Env<WIDTH, HEIGHT>,
    solution: &mut Solution,
    q_values: &mut QValues,
    opponent: &mut Opponent,
    valid_actions: &[usize],
    side: SolutionSide,
) -> usize {
    if env.current_player == side.player() {
        solution.select_best_action(env.get_state(), q_values) as usize
    } else {
        opponent.choose_action(valid_actions)
    }
}

fn random_stats(side: SolutionSide) -> OutcomeStats {
    let mut stats = OutcomeStats::default();

    for seed in 0..RANDOM_GAMES as u64 {
        let mut opponent = Opponent::Random(StdRng::seed_from_u64(seed));
        stats.record(play_solution_game(&mut opponent, side).outcome, side);
    }

    stats
}

fn print_random_stats(side: SolutionSide, stats: &OutcomeStats) {
    println!(
        "{} against random over {RANDOM_GAMES} games: {} wins, {} draws, {} losses",
        side.label(),
        stats.wins,
        stats.draws,
        stats.losses
    );
}

#[test]
#[ignore]
fn solution_returns_legal_actions_as_red() {
    let mut opponent = Opponent::Random(StdRng::seed_from_u64(0));
    play_solution_game(&mut opponent, SolutionSide::Red);
}

#[test]
#[ignore]
fn solution_returns_legal_actions_as_yellow() {
    let mut opponent = Opponent::Random(StdRng::seed_from_u64(0));
    play_solution_game(&mut opponent, SolutionSide::Yellow);
}

#[test]
#[ignore]
fn solution_runs_against_random_opponent_as_red() {
    let stats = random_stats(SolutionSide::Red);
    print_random_stats(SolutionSide::Red, &stats);
}

#[test]
#[ignore]
fn solution_runs_against_random_opponent_as_yellow() {
    let stats = random_stats(SolutionSide::Yellow);
    print_random_stats(SolutionSide::Yellow, &stats);
}

#[test]
#[ignore]
fn solution_reports_random_policy_side_bias() {
    let red = random_stats(SolutionSide::Red);
    let yellow = random_stats(SolutionSide::Yellow);

    print_random_stats(SolutionSide::Red, &red);
    print_random_stats(SolutionSide::Yellow, &yellow);

    println!(
        "side-bias delta: red losses {}, yellow losses {}",
        red.losses, yellow.losses
    );
}

#[test]
#[ignore]
fn solution_runs_against_fixed_column_opponents() {
    for side in [SolutionSide::Red, SolutionSide::Yellow] {
        run_fixed_column_opponents(side);
    }
}

fn run_fixed_column_opponents(side: SolutionSide) {
    for column in 0..WIDTH {
        let mut opponent = Opponent::FixedColumn(column);
        let report = play_solution_game(&mut opponent, side);

        println!(
            "{} fixed-column opponent {column}: {:?} in {} moves",
            side.label(),
            report.outcome,
            report.moves
        );
    }
}
