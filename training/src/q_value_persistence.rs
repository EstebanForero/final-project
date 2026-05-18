
use std::{collections::HashMap, fs::File, io::{BufReader, BufWriter, ErrorKind, Read, Write}, thread};

use crate::{connect4::Connect4Env, types::{Action, ActionQValue, CurrentState, HEIGHT, Player, QValue, QValues, State, WIDTH}};

#[derive(Clone, Copy)]
struct QExportRecord {
    state_key: u128,
    action: Action,
    q_value: QValue,
    visits: u32,
}

fn encode_state(state: &State) -> u128 {
    let a = state.get_player_a_bits() as u128;
    let b = state.get_player_b_bits() as u128;

    assert!(
        b < (1u128 << 63),
        "Cannot encode state: player B board uses bit 63, but bit 127 is reserved for current_player"
    );

    let player = match state.current_player {
        crate::types::Player::A => 0u128,
        crate::types::Player::B => 1u128,
    };

    a | (b << 64) | (player << 127)

}

fn decode_state(encoded_state: u128) -> State {
    let player_a_bits = encoded_state as u64;
    let b_mask = (1u128 << 63) - 1;
    let player_b_bits = ((encoded_state >> 64) & b_mask) as u64;

    let player_bit = (encoded_state >> 127) & 1;

    let current_player =  match player_bit {
        0 => Player::A,
        1 => Player::B,
        _ => unreachable!(),
    };

    let current_state = if Connect4Env::<WIDTH, HEIGHT>::is_win(player_a_bits) {
        CurrentState::Win(Player::A)
    } else if Connect4Env::<WIDTH, HEIGHT>::is_win(player_b_bits) {
        CurrentState::Win(Player::B)
    } else if Connect4Env::<WIDTH, HEIGHT>::is_board_full(player_a_bits | player_b_bits) {
        CurrentState::Draw
    } else {
        CurrentState::Ongoing
    };

    State::new(player_a_bits, player_b_bits, current_player, current_state)
}

fn q_values_to_q_export_record(q_values: QValues) -> Vec<QExportRecord> {
    let mut export_records = Vec::with_capacity(q_values.table.len() * 8);
    for (state, action_value) in q_values.table {
        for (action, value) in action_value {
            export_records.push(QExportRecord {
                state_key: encode_state(&state),
                action,
                q_value: value.get_q_value(),
                visits: value.get_visits()
            });
        }
    }

    export_records
}

pub struct QValuePersistence {
    path: String
}

impl QValuePersistence {
    pub fn new(path: String) -> Self {

        if !std::path::Path::new(&path).exists() {
            File::create(&path)
            .expect("Failed to create q-values first");
        }

        Self {
            path
        }
    } 

    pub fn save_q_values_background(&self, q_values: QValues) -> thread::JoinHandle<()>{
        let path = self.path.clone();

        thread::spawn(move || {
            Self::save_q_values_blocking(path, q_values);
        })
    }

    pub fn load_q_values(&self) -> QValues {
        let file = File::open(&self.path)
            .expect("Failed to open q-values file");

        let mut reader = BufReader::new(file);

        let mut q_values = HashMap::<State, HashMap<Action, ActionQValue>>::new();

        loop {
            let mut state_key_bytes = [0u8; 16];

            match reader.read_exact(&mut state_key_bytes) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(error) => {
                    panic!("Failed to read state key: {error}");
                }
            }

            let mut action_bytes = [0u8; 1];
            reader
                .read_exact(&mut action_bytes)
                .expect("Failed to read action");

            let mut q_value_bytes = [0u8; std::mem::size_of::<QValue>()];
            reader
                .read_exact(&mut q_value_bytes)
                .expect("Failed to read q-value");

            let mut visits_bytes = [0u8; 4];
            reader
                .read_exact(&mut visits_bytes)
                .expect("Failed to read visits");

            let state_key = u128::from_le_bytes(state_key_bytes);
            let state = decode_state(state_key);

            let action = action_bytes[0];

            let q_value = QValue::from_le_bytes(q_value_bytes);
            let visits = u32::from_le_bytes(visits_bytes);

            q_values.entry(state).or_default().insert(action, ActionQValue::from(q_value, visits));
        }

        QValues {
            table: q_values
        }
    }

    fn save_q_values_blocking(path: String, q_values: QValues) {
        let export_records = q_values_to_q_export_record(q_values);

        let temp_path = format!("{path}.tmp");

        let file = File::create(&temp_path)
            .expect("Failed to create temporal q-values file");

        let mut writer = BufWriter::new(file);

        for record in export_records {
            writer.write_all(&record.state_key.to_le_bytes())
                .expect("Failed to write state key");

            writer.write_all(&[record.action])
                .expect("Failed to write state key");

            writer
                .write_all(&record.q_value.to_le_bytes())
                .expect("Failed to write q-value");

            writer
                .write_all(&record.visits.to_le_bytes())
                .expect("Failed to write visits");
                
        }

        writer.flush()
            .expect("Failed to write q-values file");

        std::fs::rename(&temp_path, &path)
            .expect("Failed to replace q-values file");
    }
}
