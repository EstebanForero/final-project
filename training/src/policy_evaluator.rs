use std::collections::HashSet;

use crate::types::{ActionQValue, QValues, Transition};


pub trait PolicyTrialEvaluator {
    fn evaluate_trial(
        &self,
        trial: &[Transition],
        q_values: &mut QValues
    );
}

pub struct FirstVisitMonteCarloEvaluator {
    pub gamma: f32
}

impl PolicyTrialEvaluator for FirstVisitMonteCarloEvaluator {
    fn evaluate_trial(
        &self,
        trial: &[Transition],
        q_values: &mut QValues
    ) {
        let mut u = trial[trial.len() - 1].reward;

        let mut trial_visited = HashSet::with_capacity(44);

        for t in (0..(trial.len() - 1)).rev() {
            let reward = trial[t].reward;
            let state = &trial[t].state;
            let action = trial[t].action;

            u = self.gamma * u + reward;

            if trial_visited.contains(&(state.clone(), action)) {
                continue
            }

            if let None = q_values.get(&state, &action) {
                q_values.insert(state.clone(), action, ActionQValue::new());
            }

            q_values.get_mut(&state, &action)
                .expect("Unrechable point")
                .update(u);

            trial_visited.insert((state.clone(), action));
        }
    }
}
