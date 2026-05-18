use crate::{
    connect4::Connect4Env,
    policy_improver::Policy,
    types::{self, Action, CurrentState, Player, QValues, State},
};

pub struct TransitionResult {
    pub state: State,
    pub reward: f32,
}

pub struct SelfPlayEnvironment<P> {
    policy: P,
    connect4: Connect4Env<{ types::WIDTH }, { types::HEIGHT }>,
}

impl<P: Policy> SelfPlayEnvironment<P> {
    pub fn new(policy: P) -> Self {
        Self {
            policy,
            connect4: Connect4Env::new(),
        }
    }

    pub fn from_state(policy: P, initial_state: State) -> Self {
        Self {
            policy,
            connect4: Connect4Env::from(initial_state),
        }
    }

    pub fn get_current_state(&self) -> State {
        self.connect4.get_state()
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        self.connect4
            .valid_actions()
            .into_iter()
            .map(|action| action as u8)
            .collect()
    }

    pub fn transition(&mut self, action: Action, q_values: &QValues) -> TransitionResult {
        let current_player = self.connect4.current_player;

        self.connect4.play_move(action as usize);
        let new_state = self.connect4.get_state();

        if new_state.current_state != CurrentState::Ongoing {
            let reward = reward_for_player(&new_state, current_player);

            return TransitionResult {
                state: new_state,
                reward,
            };
        }

        let other_player_actions: Vec<Action> = self
            .connect4
            .valid_actions()
            .into_iter()
            .map(|x| x as u8)
            .collect();

        let opponent_action =
            self.policy
                .choose_action(&new_state, &other_player_actions, q_values);

        self.connect4.play_move(opponent_action as usize);

        let new_state = self.connect4.get_state();

        let reward = reward_for_player(&new_state, current_player);

        TransitionResult {
            state: new_state,
            reward: reward,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.connect4.current_state != CurrentState::Ongoing
    }
}

pub fn reward_for_player(state: &State, player: Player) -> f32 {
    match state.current_state {
        CurrentState::Win(winner) if winner == player => 1.0,
        CurrentState::Win(_) => -1.0,
        CurrentState::Ongoing => 0.0,
        CurrentState::Draw => 0.0,
    }
}
