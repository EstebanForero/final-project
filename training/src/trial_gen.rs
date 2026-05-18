use crate::{policy_improver::Policy, types::{State, Transition}};


pub trait TrialGenerator {
    fn generate_trial(
        &mut self,
        initial_state: State,
    ) -> Vec<Transition>;
}

pub struct MonteCarloTrialGenerator<T, R> {
    tree_policy: T,
    rollout_policy: R
}

impl<T: Policy, R: Policy>  MonteCarloTrialGenerator<T, R> {
    pub fn new(
        tree_policy: T,
        rollout_policy: R
    ) -> Self {
        Self {
            tree_policy,
            rollout_policy
        }
    }
}

impl<T: Policy, R: Policy> TrialGenerator for MonteCarloTrialGenerator<T, R> {
    fn generate_trial(
        &mut self,
        initial_state: State,
    ) -> Vec<Transition> {
        todo!()
    }
}
