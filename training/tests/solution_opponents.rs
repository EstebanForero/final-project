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

#[derive(Default)]
struct OutcomeStats {
    wins: usize,
    draws: usize,
    losses: usize,
}

impl OutcomeStats {
    fn record(&mut self, outcome: CurrentState) {
        match outcome {
            CurrentState::Win(Player::A) => self.wins += 1,
            CurrentState::Win(Player::B) => self.losses += 1,
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

fn play_solution_game(opponent: &mut Opponent) -> GameReport {
    let mut env = Connect4Env::<WIDTH, HEIGHT>::new();
    let mut q_values = load_q_values();
    let mut solution = build_solution();

    for moves in 0..(WIDTH * HEIGHT) {
        if let Some(report) = finished_report(&env, moves) {
            return report;
        }

        play_turn(&mut env, &mut solution, &mut q_values, opponent);
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
) {
    let valid_actions = env.valid_actions();
    let action = choose_action(env, solution, q_values, opponent, &valid_actions);

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
) -> usize {
    if env.current_player == Player::A {
        solution.select_best_action(env.get_state(), q_values) as usize
    } else {
        opponent.choose_action(valid_actions)
    }
}

#[test]
#[ignore]
fn solution_runs_against_random_opponent() {
    let mut stats = OutcomeStats::default();

    for seed in 0..RANDOM_GAMES as u64 {
        let mut opponent = Opponent::Random(StdRng::seed_from_u64(seed));
        stats.record(play_solution_game(&mut opponent).outcome);
    }

    println!(
        "random opponent results over {RANDOM_GAMES} games: {} wins, {} draws, {} losses",
        stats.wins, stats.draws, stats.losses
    );
}

#[test]
#[ignore]
fn solution_runs_against_fixed_column_opponents() {
    for column in 0..WIDTH {
        let mut opponent = Opponent::FixedColumn(column);
        let report = play_solution_game(&mut opponent);

        println!(
            "fixed-column opponent {column}: {:?} in {} moves",
            report.outcome, report.moves
        );
    }
}
