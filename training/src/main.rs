use crate::q_value_persistence::QValuePersistence;

pub mod connect4;
pub mod types;
pub mod q_value_persistence;
pub mod policy_evaluator;
pub mod policy_improver;
pub mod trial_gen;
pub mod alternating_markov_games;

fn main() {
    let q_value_persistence = QValuePersistence::new("./q_values".into());
    let global_q_values = q_value_persistence.load_q_values();
}
