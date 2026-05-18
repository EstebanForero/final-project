use std::collections::HashSet;

use crate::types::{ActionQValue, QValues, Transition};

pub trait PolicyTrialEvaluator {
    fn evaluate_trial(&self, trial: &[Transition], q_values: &mut QValues);
}

pub struct FirstVisitMonteCarloEvaluator {
    pub gamma: f32,
}

impl FirstVisitMonteCarloEvaluator {
    pub fn new(gamma: f32) -> Self {
        Self { gamma }
    }
}

impl PolicyTrialEvaluator for FirstVisitMonteCarloEvaluator {
    fn evaluate_trial(&self, trial: &[Transition], q_values: &mut QValues) {
        let mut u = trial[trial.len() - 1].reward;

        let mut trial_visited = HashSet::with_capacity(44);

        for t in (0..(trial.len() - 1)).rev() {
            let reward = trial[t].reward;
            let state = &trial[t].state;
            let action = trial[t].action;

            u = self.gamma * u + reward;

            if trial_visited.contains(&(state.clone(), action)) {
                continue;
            }

            if let None = q_values.get(&state, &action) {
                q_values.insert(state.clone(), action, ActionQValue::new());
            }

            q_values
                .get_mut(&state, &action)
                .expect("Unrechable point")
                .update(u);

            trial_visited.insert((state.clone(), action));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        policy_evaluator::{FirstVisitMonteCarloEvaluator, PolicyTrialEvaluator},
        types::{CurrentState, Player, QValues, State, Transition},
    };

    #[test]
    #[ignore]
    fn test_single_terminal_transition_updates_q_value() {
        let state = State::new(0, 0, Player::A, CurrentState::Ongoing);
        let trial = vec![Transition {
            state: state.clone(),
            action: 3,
            reward: 1.0,
        }];
        let mut q_values = QValues {
            table: HashMap::new(),
        };

        FirstVisitMonteCarloEvaluator::new(1.0).evaluate_trial(&trial, &mut q_values);

        let value = q_values
            .get(&state, &3)
            .expect("terminal transition should be learned");

        assert_eq!(value.get_q_value(), 1.0);
        assert_eq!(value.get_visits(), 1);
    }

    fn q_values() -> QValues {
        QValues {
            table: HashMap::new(),
        }
    }

    fn state(player: Player) -> State {
        State::new(0, 0, player, CurrentState::Ongoing)
    }

    fn two_step_trial(player: Player, action: u8) -> Vec<Transition> {
        vec![
            Transition {
                state: state(player),
                action,
                reward: 0.0,
            },
            Transition {
                state: state(player.other()),
                action: action + 1,
                reward: 1.0,
            },
        ]
    }

    #[test]
    fn test_two_step_trials_update_player_a_and_player_b_equally() {
        let evaluator = FirstVisitMonteCarloEvaluator::new(1.0);
        let player_a_state = state(Player::A);
        let player_b_state = state(Player::B);
        let mut player_a_q_values = q_values();
        let mut player_b_q_values = q_values();

        evaluator.evaluate_trial(&two_step_trial(Player::A, 2), &mut player_a_q_values);
        evaluator.evaluate_trial(&two_step_trial(Player::B, 2), &mut player_b_q_values);

        let player_a_value = player_a_q_values.get(&player_a_state, &2).unwrap();
        let player_b_value = player_b_q_values.get(&player_b_state, &2).unwrap();

        assert_eq!(player_a_value.get_q_value(), player_b_value.get_q_value());
        assert_eq!(player_a_value.get_visits(), player_b_value.get_visits());
    }
}
