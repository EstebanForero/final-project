use crate::{policy_evaluator::{FirstVisitMonteCarloEvaluator, PolicyTrialEvaluator}, policy_improver::{GreedyPolicy, UcbPolicy}, q_value_persistence::QValuePersistence, trial_gen::{MonteCarloTreeSearch, OnlinePolicyImprovementTrialGenerator, TrialGenerator}};
use crate::trial_gen::ProjectedTrial;

pub mod connect4;
pub mod types;
pub mod q_value_persistence;
pub mod policy_evaluator;
pub mod policy_improver;
pub mod trial_gen;
pub mod alternating_markov_games;

fn main() {
    let q_value_persistence = QValuePersistence::new("./q_values".into());
    
    let mut global_q_values = q_value_persistence.load_q_values();

    let tree_policy = UcbPolicy::new(1.4);
    let rollout_policy = GreedyPolicy::new();
    let montecarlo_tree_search = MonteCarloTreeSearch::new(tree_policy, rollout_policy, 1000);
    let mut trial_generator = OnlinePolicyImprovementTrialGenerator::new(montecarlo_tree_search, global_q_values.clone());

    let policy_evaluator = FirstVisitMonteCarloEvaluator::new(1.);

    for iteration in 0..10_000 {
        trial_generator.set_q_values(global_q_values.clone());
        let (trial_player_a, trial_player_b) = trial_generator.generate_trial().project_for_players();

        policy_evaluator.evaluate_trial(&trial_player_a, &mut global_q_values);
        policy_evaluator.evaluate_trial(&trial_player_b, &mut global_q_values);

        if iteration % 1000 == 0 {
            println!("Iteration: {iteration} \n QValues: {}", global_q_values.table.len());
            println!("Saving q values");
        }
    }
}
