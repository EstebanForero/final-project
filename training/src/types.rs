use std::collections::HashMap;

pub const WIDTH: usize = 7;
pub const HEIGHT: usize = 6;

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

    pub fn get_player_a_bits(&self) -> u64 {
        self.player_a_bits
    }

    pub fn get_player_b_bits(&self) -> u64 {
        self.player_b_bits
    }
}

pub type Action = u8;

pub type QValue = f32;

#[derive(Clone)]
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

    pub fn from(q_value: QValue, visits: u32) -> Self {
        Self {
            q_value,
            visits
        }
    }

    pub fn update(&mut self, observed_return: f32) {
        self.visits += 1;
        self.q_value += (observed_return - self.q_value) / self.visits as f32
    }

    pub fn get_q_value(&self) -> QValue {
        return self.q_value
    }

    pub fn get_visits(&self) -> u32 {
        return self.visits
    }
}

pub struct QValues {
    pub table: HashMap<State, HashMap<Action, ActionQValue>>
}

impl QValues {
    pub fn get(&self, state: &State, action: &Action) -> Option<ActionQValue> {
        Some(self.table.get(state)?.get(action)?.clone())
    }
}
