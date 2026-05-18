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

impl GreedyPolicy {
    pub fn new() -> Self {
        Self {}
    }
}

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

pub struct UcbPolicy {
    pub exploration_c: f32
}

fn ucb_score(exploration_c: f32, q_value: f32, action_visits: u32, total_visits: u32) -> f32 {
    if action_visits == 0 {
        return f32::INFINITY
    }

    let total_visits = total_visits as f32;
    let action_visits = action_visits as f32;
    q_value + exploration_c * (total_visits.ln() / action_visits).sqrt()
}

impl Policy for UcbPolicy {
    fn choose_action(
        &self,
        state: &State,
        valid_actions: &[Action],
        q_values: &QValues,
    ) -> Action {
        assert!(
            !valid_actions.is_empty(),
            "UcbPolicy cannot choose action: no valid actions available"
        );

        let total_visits: u32 = valid_actions
            .iter()
            .map(|action| q_values.get(state, action).unwrap_or_default().get_visits())
            .sum();

        let action = valid_actions
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let a_value = q_values.get(state, &a).unwrap_or_default();
                let b_value = q_values.get(state, &b).unwrap_or_default();

                let a_score = ucb_score(
                    self.exploration_c,
                    a_value.get_q_value(),
                    a_value.get_visits(),
                    total_visits,
                );

                let b_score = ucb_score(
                    self.exploration_c,
                    b_value.get_q_value(),
                    b_value.get_visits(),
                    total_visits,
                );

                a_score.total_cmp(&b_score)
            })
            .unwrap();

        action
    }
}
