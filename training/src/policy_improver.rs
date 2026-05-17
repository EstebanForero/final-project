use rand::seq::IndexedRandom;

use crate::types::{Action, ActionQValue, QValues, State};


pub trait Policy {
    fn choose_action(
        &self,
        state: &State,
        valid_actions: &[Action],
        q_values: &QValues,
    ) -> Action;
}

pub struct GreedyPolicy {}

impl Policy for GreedyPolicy {
    fn choose_action(
        &self,
        state: &State,
        valid_actions: &[Action],
        q_values: &QValues,
    ) -> Action {
        assert!(
            !valid_actions.is_empty(),
            "GreedyPolicy cannot choose action: no valid actions available"
        );

        let best_q = valid_actions
            .iter()
            .copied()
            .map(|action| q_values
                .get(state, &action)
                .unwrap_or(ActionQValue::new())
                .get_q_value()
            )
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        let best_actions: Vec<Action> = valid_actions
            .iter()
            .copied()
            .filter(|action| {
                q_values
                    .get(state, action)
                    .unwrap_or(ActionQValue::new())
                    .get_q_value()
                    == best_q
            })
            .collect();

        *best_actions
            .choose(&mut rand::rng())
            .expect("No best action available")
    }
}
