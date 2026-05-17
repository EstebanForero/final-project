use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    A,
    B,
}

impl Player {
    pub fn other(&self) -> Player {
        match self {
            Player::A => Player::B,
            Player::B => Player::A,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurrentState {
    Ongoing,
    Win(Player),
    Draw
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct State {
    player_a_bits: u64,
    player_b_bits: u64,
    pub current_player: Player,
    pub current_state: CurrentState
}

impl State {
    pub fn new(player_a_bits: u64, player_b_bits: u64, current_player: Player, current_state: CurrentState) -> Self {
        Self {
            player_a_bits,
            player_b_bits,
            current_player,
            current_state
        }
    }
}

pub type Action = u8;

pub type QValue = f32;

pub struct ActionQValue {
    q_value: QValue,
    visits: u32
}

impl ActionQValue {
    pub fn new() -> Self {
        Self {
            q_value: 0.0,
            visits: 0
        }
    }

    pub fn update(&mut self, observed_return: f32) {
        self.visits += 1;
        self.q_value += (observed_return - self.q_value) / self.visits as f32
    }

    pub fn get_q_value(&self) -> QValue {
        return self.q_value
    }
}

pub struct QValues {
    table: HashMap<State, HashMap<Action, ActionQValue>>
}


