use std::thread::JoinHandle;

use rayon::prelude::*;

use training::trial_gen::ProjectedTrial;
use training::types::Transition;
use training::{
    policy_evaluator::{FirstVisitMonteCarloEvaluator, PolicyTrialEvaluator},
    policy_improver::{GreedyPolicy, UcbPolicy},
    q_value_persistence::QValuePersistence,
    trial_gen::{MonteCarloTreeSearch, OnlinePolicyImprovementTrialGenerator, TrialGenerator},
};

const BATCH_SIZE: usize = 8;

fn make_trial_generator() -> OnlinePolicyImprovementTrialGenerator<UcbPolicy, GreedyPolicy> {
    let tree_policy = UcbPolicy::new(1.4);
    let rollout_policy = GreedyPolicy::new();

    let montecarlo_tree_search = MonteCarloTreeSearch::new(tree_policy, rollout_policy, 100);

    OnlinePolicyImprovementTrialGenerator::new(montecarlo_tree_search)
}

fn main() {
    let q_value_persistence = QValuePersistence::new("./q_values".into());

    let mut global_q_values = q_value_persistence.load_q_values();

    let policy_evaluator = FirstVisitMonteCarloEvaluator::new(1.0);

    let mut save_handle: Option<JoinHandle<()>> = None;

    for iteration in 0..10_000_000 {
        let projected_trials: Vec<(Vec<Transition>, Vec<Transition>)> = (0..BATCH_SIZE)
            .into_par_iter()
            .map(|_| {
                let mut trial_generator = make_trial_generator();

                trial_generator
                    .generate_trial(&global_q_values)
                    .project_for_players()
            })
            .collect();

        for (trial_player_a, trial_player_b) in projected_trials {
            policy_evaluator.evaluate_trial(&trial_player_a, &mut global_q_values);
            policy_evaluator.evaluate_trial(&trial_player_b, &mut global_q_values);
        }

        if iteration % 50_000 == 0 && iteration != 0 {
            println!(
                "Iteration: {iteration}\nQValues: {}",
                global_q_values.table.len()
            );
        }

        if iteration % 200_000 == 0 && iteration != 0 {
            println!(
                "Iteration: {iteration}\nQValues: {}",
                global_q_values.table.len()
            );

            println!("Saving q values");

            if let Some(handle) = save_handle.take() {
                handle
                    .join()
                    .expect("Previous q-values save thread panicked");
            }

            save_handle =
                Some(q_value_persistence.save_q_values_background(global_q_values.clone()));
        }
    }

    if let Some(handle) = save_handle {
        handle.join().expect("Final q-values save thread panicked");
    }

    q_value_persistence
        .save_q_values_background(global_q_values)
        .join()
        .expect("Final q-values save thread panicked");
}
