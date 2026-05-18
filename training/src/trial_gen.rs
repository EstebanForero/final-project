use crate::{alternating_markov_games::SelfPlayEnvironment, policy_improver::{GreedyPolicy, Policy}, types::{QValues, State, Transition}};


pub trait TrialGenerator {
    fn generate_trial(
        &mut self,
        initial_state: State,
    ) -> Vec<Transition>;
}

pub struct OnlinePolicyImprovementTrialGenerator<T, R> {
    local_search: MonteCarloTreeSearch<T, R>,
    inner_q_values: QValues
}

impl<T: Policy, R: Policy> OnlinePolicyImprovementTrialGenerator<T, R> {
    pub fn new(local_search: MonteCarloTreeSearch<T, R>, global_q_values: QValues) -> Self {
        Self {
            local_search,
            inner_q_values: global_q_values
        }
    }
}

impl<T: Policy, R: Policy> TrialGenerator for OnlinePolicyImprovementTrialGenerator<T, R> {

    fn generate_trial(
        &mut self,
        initial_state: State,
    ) -> Vec<Transition> {
        let greedy_policy = GreedyPolicy::new();
        let self_play_env = SelfPlayEnvironment::from_state(greedy_policy, initial_state);

            todo!()
    }
}

pub struct MonteCarloTreeSearch<T, R> {
    tree_policy: T,
    rollout_policy: R
}
