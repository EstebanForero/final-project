use std::collections::HashSet;

use crate::{
    alternating_markov_games::{SelfPlayEnvironment, reward_for_player},
    connect4::Connect4Env,
    policy_improver::Policy,
    types::{
        self, Action, Player, QValueStoreRead, QValueStoreWrite, QValues, QValuesOverlay, State,
        Transition,
    },
};

pub trait ProjectedTrial {
    fn project_for_players(&self) -> (Vec<Transition>, Vec<Transition>);
}

impl ProjectedTrial for Vec<Transition> {
    fn project_for_players(&self) -> (Vec<Transition>, Vec<Transition>) {
        let mut player_a_trial = Vec::new();
        let mut player_b_trial = Vec::new();

        for transition in self {
            match transition.state.current_player {
                Player::A => player_a_trial.push(transition.clone()),
                Player::B => player_b_trial.push(transition.clone()),
            }
        }

        let last = self.last().expect("Trial shouldn't be empty");
        if last.state.current_player == Player::A {
            player_b_trial
                .last_mut()
                .expect("Trial player b shouldn't be empty")
                .reward = -last.reward;
        } else {
            player_a_trial
                .last_mut()
                .expect("Trial player a shouldn't be empty")
                .reward = -last.reward;
        }

        (player_a_trial, player_b_trial)
    }
}

pub trait TrialGenerator {
    fn generate_trial(&mut self, global_q_values: &QValues) -> Vec<Transition>;
}

pub struct OnlinePolicyImprovementTrialGenerator<T, R> {
    local_search: MonteCarloTreeSearch<T, R>,
}

impl<T: Policy, R: Policy> OnlinePolicyImprovementTrialGenerator<T, R> {
    pub fn new(local_search: MonteCarloTreeSearch<T, R>) -> Self {
        Self { local_search }
    }
}

impl<T: Policy, R: Policy> TrialGenerator for OnlinePolicyImprovementTrialGenerator<T, R> {
    fn generate_trial(&mut self, global_q_values: &QValues) -> Vec<Transition> {
        let mut q_values = QValuesOverlay::new(global_q_values);
        let mut connect4_environment = Connect4Env::<{ types::WIDTH }, { types::HEIGHT }>::new();

        let mut transitions = Vec::new();

        loop {
            let current_state = connect4_environment.get_state();

            if current_state.is_terminal() {
                break;
            }

            let action = self
                .local_search
                .select_best_action(current_state.clone(), &mut q_values);
            connect4_environment.play_move(action as usize);

            let next_state = connect4_environment.get_state();
            let player = connect4_environment.current_player.other();
            let reward_current_player = reward_for_player(&next_state, player);

            transitions.push(Transition {
                state: current_state,
                action,
                reward: reward_current_player,
            });
        }

        transitions
    }
}

pub struct MonteCarloTreeSearch<T, R> {
    tree_policy: T,
    rollout_policy: R,
    simulation_budget: usize,
}

impl<T: Policy, R: Policy> MonteCarloTreeSearch<T, R> {
    pub fn new(tree_policy: T, rollout_policy: R, simulation_budget: usize) -> Self {
        Self {
            tree_policy,
            rollout_policy,
            simulation_budget,
        }
    }

    pub fn select_best_action<Q>(&mut self, root_state: State, q_values: &mut Q) -> Action
    where
        Q: QValueStoreWrite,
    {
        let self_play_env =
            SelfPlayEnvironment::from_state(self.rollout_policy.clone(), root_state);

        let root_state = self_play_env.get_current_state();
        let root_valid_actions = self_play_env.valid_actions();

        let mut arena_tree = ArenaTree::new(root_state.clone(), root_valid_actions.clone());

        for _ in 0..self.simulation_budget {
            self.run_simulation(&mut arena_tree, q_values);
        }

        self.tree_policy
            .choose_action(&root_state, &root_valid_actions, q_values)
    }

    fn run_simulation<Q>(&self, arena_tree: &mut ArenaTree, q_values: &mut Q)
    where
        Q: QValueStoreWrite,
    {
        let selected_node = self.selection(arena_tree, q_values);

        if arena_tree.get_node(selected_node).state.is_terminal() {
            return;
        }

        let expanded_node = self.expansion(selected_node, arena_tree, q_values);

        let return_in_expanded_node = self.rollout(expanded_node, arena_tree, q_values);

        self.backtracking(expanded_node, arena_tree, q_values, return_in_expanded_node);
    }

    fn backtracking<Q>(
        &self,
        expanded_node_id: NodeId,
        arena_tree: &ArenaTree,
        q_values: &mut Q,
        expanded_node_return: f32,
    ) where
        Q: QValueStoreWrite,
    {
        let mut current_node_id = expanded_node_id;

        loop {
            let current_node = arena_tree.get_node(current_node_id);

            let Some(parent_id) = current_node.parent else {
                return;
            };

            let Some(action) = current_node.action_from_parent else {
                return;
            };

            let parent_node = arena_tree.get_node(parent_id);

            q_values.update(&parent_node.state, &action, expanded_node_return);

            current_node_id = parent_id;
        }
    }

    fn rollout<Q>(&self, expanded_node_id: NodeId, arena_tree: &mut ArenaTree, q_values: &Q) -> f32
    where
        Q: QValueStoreRead,
    {
        let expanded_node = arena_tree.get_node(expanded_node_id);

        let mut self_play_env = SelfPlayEnvironment::from_state(
            self.rollout_policy.clone(),
            expanded_node.state.clone(),
        );
        let mut last_reward = 0.;

        loop {
            let valid_actions = self_play_env.valid_actions();
            if self_play_env.valid_actions().is_empty() || self_play_env.is_terminal() {
                return last_reward;
            }

            let current_state = self_play_env.get_current_state();
            let action =
                self.rollout_policy
                    .choose_action(&current_state, &valid_actions, q_values);
            let transition_result = self_play_env.transition(action, q_values);
            last_reward = transition_result.reward;
        }
    }

    fn expansion<Q>(
        &self,
        selected_node_id: NodeId,
        arena_tree: &mut ArenaTree,
        q_values: &Q,
    ) -> NodeId
    where
        Q: QValueStoreRead,
    {
        let selected_node = arena_tree.get_node(selected_node_id);
        let action = selected_node.untried_actions.iter().next()
            .expect("It shouldn't fail since we checked that there were untried actions in the selected node, and terminal state check should be done outside of expansion");

        let mut self_play_env = SelfPlayEnvironment::from_state(
            self.rollout_policy.clone(),
            selected_node.state.clone(),
        );
        let result = self_play_env.transition(*action, q_values);

        arena_tree.add_child(
            selected_node_id,
            *action,
            result.state,
            self_play_env.valid_actions(),
        )
    }

    fn selection<Q>(&self, arena_tree: &ArenaTree, q_values: &Q) -> NodeId
    where
        Q: QValueStoreRead,
    {
        let mut current_node_id = arena_tree.root;

        loop {
            let current_node = arena_tree.get_node(current_node_id);
            if !current_node.untried_actions.is_empty() || current_node.children.is_empty() {
                return current_node_id;
            }

            let action = self.tree_policy.choose_action(
                &current_node.state,
                &current_node.get_child_actions(),
                q_values,
            );

            current_node_id = arena_tree
                .get_child_for_action(current_node_id, action)
                .expect("It should always exist");
        }
    }
}

type NodeId = usize;

pub struct ArenaTree {
    tree: Vec<MctsNode>,
    root: NodeId,
}

impl ArenaTree {
    pub fn new(root_state: State, root_valid_actions: Vec<Action>) -> Self {
        let root_node = MctsNode {
            state: root_state,
            parent: None,
            action_from_parent: None,

            children: Vec::new(),
            untried_actions: HashSet::from_iter(root_valid_actions.into_iter()),
        };

        Self {
            tree: vec![root_node],
            root: 0,
        }
    }

    pub fn get_root(&self) -> NodeId {
        self.root
    }

    pub fn change_root(&mut self, new_root_id: NodeId) {
        self.root = new_root_id;
        self.tree[new_root_id].parent = None;
        self.tree[new_root_id].action_from_parent = None;
    }

    pub fn get_node(&self, node_id: NodeId) -> &MctsNode {
        &self.tree[node_id]
    }

    pub fn get_child_for_action(&self, node_id: NodeId, action: Action) -> Option<NodeId> {
        self.tree[node_id]
            .children
            .iter()
            .find(|(action_c, _)| action == *action_c)
            .map(|(_, child_index)| *child_index)
    }

    pub fn get_node_mut(&mut self, node_id: NodeId) -> &mut MctsNode {
        &mut self.tree[node_id]
    }

    pub fn add_child(
        &mut self,
        parent_id: NodeId,
        action: Action,
        child_state: State,
        child_valid_actions: Vec<Action>,
    ) -> NodeId {
        let child_id = self.tree.len();

        let child_node = MctsNode {
            state: child_state,
            parent: Some(parent_id),
            action_from_parent: Some(action),

            children: Vec::new(),
            untried_actions: HashSet::from_iter(child_valid_actions.into_iter()),
        };

        self.tree.push(child_node);

        self.tree[parent_id].children.push((action, child_id));
        self.tree[parent_id].untried_actions.remove(&action);

        child_id
    }
}

pub struct MctsNode {
    pub state: State,
    pub parent: Option<NodeId>,
    pub action_from_parent: Option<Action>,

    pub children: Vec<(Action, NodeId)>,
    pub untried_actions: HashSet<Action>,
}

impl MctsNode {
    fn get_child_actions(&self) -> Vec<Action> {
        self.children.iter().map(|(action, _)| *action).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        trial_gen::ProjectedTrial,
        types::{CurrentState, Player, State, Transition},
    };

    fn state(player: Player) -> State {
        State::new(0, 0, player, CurrentState::Ongoing)
    }

    fn transition(player: Player, action: u8, reward: f32) -> Transition {
        Transition {
            state: state(player),
            action,
            reward,
        }
    }

    #[test]
    fn project_for_players_keeps_player_a_win_and_player_b_loss() {
        let trial = vec![
            transition(Player::A, 0, 0.0),
            transition(Player::B, 1, 0.0),
            transition(Player::A, 2, 1.0),
        ];

        let (player_a, player_b) = trial.project_for_players();

        assert_eq!(player_a.len(), 2);
        assert_eq!(player_b.len(), 1);
        assert_eq!(player_a[1].reward, 1.0);
        assert_eq!(player_b[0].reward, -1.0);
    }

    #[test]
    fn project_for_players_keeps_player_b_win_and_player_a_loss() {
        let trial = vec![transition(Player::A, 0, 0.0), transition(Player::B, 1, 1.0)];

        let (player_a, player_b) = trial.project_for_players();

        assert_eq!(player_a.len(), 1);
        assert_eq!(player_b.len(), 1);
        assert_eq!(player_a[0].reward, -1.0);
        assert_eq!(player_b[0].reward, 1.0);
    }
}
