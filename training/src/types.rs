use std::collections::HashMap;

pub const WIDTH: usize = 7;
pub const HEIGHT: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CurrentState {
    Ongoing,
    Win(Player),
    Draw,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct State {
    player_a_bits: u64,
    player_b_bits: u64,
    pub current_player: Player,
    pub current_state: CurrentState,
}

impl State {
    pub fn new(
        player_a_bits: u64,
        player_b_bits: u64,
        current_player: Player,
        current_state: CurrentState,
    ) -> Self {
        Self {
            player_a_bits,
            player_b_bits,
            current_player,
            current_state,
        }
    }

    pub fn get_player_a_bits(&self) -> u64 {
        self.player_a_bits
    }

    pub fn get_player_b_bits(&self) -> u64 {
        self.player_b_bits
    }

    pub fn is_terminal(&self) -> bool {
        self.current_state != CurrentState::Ongoing
    }
}

pub type Action = u8;

pub type Reward = f32;

pub type QValue = f32;

#[derive(Clone, Copy, Default)]
pub struct ActionQValue {
    q_value: QValue,
    visits: u32,
}

impl ActionQValue {
    pub fn new() -> Self {
        Self {
            q_value: 0.0,
            visits: 0,
        }
    }

    pub fn from(q_value: QValue, visits: u32) -> Self {
        Self { q_value, visits }
    }

    pub fn update(&mut self, observed_return: f32) {
        self.visits += 1;
        self.q_value += (observed_return - self.q_value) / self.visits as f32
    }

    pub fn get_q_value(&self) -> QValue {
        return self.q_value;
    }

    pub fn get_visits(&self) -> u32 {
        return self.visits;
    }
}

#[derive(Clone)]
pub struct QValues {
    pub table: HashMap<State, [ActionQValue; WIDTH]>,
}

impl QValues {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn get_mut(&mut self, state: &State, action: &Action) -> Option<&mut ActionQValue> {
        Some(self.table.get_mut(state)?.get_mut(*action as usize)?)
    }

    pub fn get(&self, state: &State, action: &Action) -> Option<ActionQValue> {
        Some(self.table.get(state)?.get(*action as usize)?.clone())
    }

    pub fn insert(&mut self, state: State, action: Action, action_q_value: ActionQValue) {
        let actions = self
            .table
            .entry(state)
            .or_insert([ActionQValue::default(); WIDTH]);

        actions[action as usize] = action_q_value
    }
}

impl Default for QValues {
    fn default() -> Self {
        Self::new()
    }
}

pub trait QValueStoreRead {
    fn get(&self, state: &State, action: &Action) -> Option<ActionQValue>;
}

pub trait QValueStoreWrite: QValueStoreRead {
    fn get_mut_or_insert(&mut self, state: &State, action: &Action) -> &mut ActionQValue;

    fn update(&mut self, state: &State, action: &Action, observed_return: f32) {
        self.get_mut_or_insert(state, action)
            .update(observed_return);
    }
}

impl QValueStoreRead for QValues {
    fn get(&self, state: &State, action: &Action) -> Option<ActionQValue> {
        QValues::get(self, state, action)
    }
}

impl QValueStoreWrite for QValues {
    fn get_mut_or_insert(&mut self, state: &State, action: &Action) -> &mut ActionQValue {
        if self.get(state, action).is_none() {
            self.insert(state.clone(), *action, ActionQValue::new());
        }

        self.get_mut(state, action)
            .expect("Value was inserted into Q-values")
    }
}

pub struct QValuesOverlay<'a> {
    global: &'a QValues,
    local: QValues,
}

impl<'a> QValuesOverlay<'a> {
    pub fn new(global: &'a QValues) -> Self {
        Self {
            global,
            local: QValues::new(),
        }
    }

    pub fn local(&self) -> &QValues {
        &self.local
    }

    pub fn into_local(self) -> QValues {
        self.local
    }
}

impl<'a> QValueStoreRead for QValuesOverlay<'a> {
    fn get(&self, state: &State, action: &Action) -> Option<ActionQValue> {
        if let Some(local_value) = self.local.get(state, action) {
            return Some(local_value);
        }

        self.global.get(state, action)
    }
}

impl<'a> QValueStoreWrite for QValuesOverlay<'a> {
    fn get_mut_or_insert(&mut self, state: &State, action: &Action) -> &mut ActionQValue {
        if self.local.get(state, action).is_none() {
            let initial_value = self
                .global
                .get(state, action)
                .unwrap_or_else(ActionQValue::new);
            self.local.insert(state.clone(), *action, initial_value);
        }

        self.local
            .get_mut(state, action)
            .expect("Value was inserted into local Q-values")
    }
}

/// ============================================= TRIALS ============================

#[derive(Clone)]
pub struct Transition {
    pub state: State,
    pub action: Action,
    pub reward: Reward,
}
