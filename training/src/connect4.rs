// const HEIGHT: usize = 6;
// const WIDTH: usize = 7;
// const STRIDE: usize = HEIGHT + 1; // Bits per column

use crate::types::{CurrentState, Player, State};

/// Our representation for a mask will be the following
/// 0 | 0 | 0 sentinel bits
/// 0 | 0 | 0
/// 0 | 0 | 0
/// 0 | 0 | 0
///
/// Example 2
///
/// 0 | 0 | 0 sentinel bits
/// 1 | 1 | 1
/// 1 | 1 | 1
/// 1 | 1 | 1
///
/// 111011101110
///
/// This for a 3x3 board
///
/// Example 3
///
/// | a  | b  | c  |
/// | a3 | b3 | c3 |
/// | a2 | b2 | c2 |
/// | a1 | b1 | c1 |
///
/// c c3 c2 c1 b b3 b2 b1 a a3 a2 a1

pub struct Connect4Env<const WIDTH: usize, const HEIGHT: usize> {
    player_a_bits: u64,
    player_b_bits: u64,
    mask: u64,
    pub current_player: Player,
    heights: [u8; WIDTH],
    pub current_state: CurrentState
}

impl<const WIDTH: usize, const HEIGHT: usize> From<State> for Connect4Env<WIDTH, HEIGHT> {
    fn from(value: State) -> Self {
        let player_a_bits = value.get_player_a_bits();
        let player_b_bits = value.get_player_b_bits();
        let mask = player_a_bits | player_b_bits;

        let heights = std::array::from_fn(|col| {
            let base = col * Self::STRIDE;

            let mut row = 0;

            while row < HEIGHT {
                let bit_index = base + row;
                let bit = 1u64 << bit_index;

                if mask & bit == 0 {
                    break;
                }

                row += 1;
            }

            (base + row) as u8
        });

        Connect4Env {
            player_a_bits,
            player_b_bits,
            current_player: value.current_player,
            mask,
            heights,
            current_state: CurrentState::Ongoing,
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> Connect4Env<WIDTH, HEIGHT> {
    const STRIDE: usize = HEIGHT + 1;

    pub fn new() -> Connect4Env<WIDTH, HEIGHT> {
        Connect4Env {
            player_a_bits: 0,
            player_b_bits: 0,
            current_player: Player::A,
            mask: 0,
            // This is an array that tells us where the next piece will fall in each column
            heights: std::array::from_fn(|col| (col * Self::STRIDE) as u8),
            current_state: CurrentState::Ongoing
        }
    }

    pub fn get_state(&self) -> State {
        State::new(self.player_a_bits, self.player_b_bits, self.current_player, self.current_state)
    }

    fn top_mask(col: usize) -> u64 {
        // We have hour board for example in the 3x3 example
        // 0 000 0 000 0 001, and we shift to the left given this formula, so we get a mask, with
        //   the top bit in 1 for the column
        1u64 << (col * Self::STRIDE + HEIGHT - 1)
    }

    fn can_play(&self, col: usize) -> bool {
        Self::can_play_helper(self.mask, col)
    }

    fn can_play_helper(mask: u64, col: usize) -> bool {
        col < WIDTH && (mask & Self::top_mask(col)) == 0
    }

    pub fn play_move(&mut self, col: usize) {
        assert!(
            self.current_state == CurrentState::Ongoing,
            "Cannot play move: game is already over"
        );
        assert!(
            self.can_play(col),
            "Invalid move: column is full or out of bounds"
        );

        

        let bit_index = self.heights[col] as usize;
        let bit = 1u64 << bit_index;

        match self.current_player {
            Player::A => self.player_a_bits |= bit,
            Player::B => self.player_b_bits |= bit,
        }

        self.mask |= bit;
        self.heights[col] += 1;

        self.update_current_state();

        self.current_player = self.current_player.other();
    }

    fn update_current_state(&mut self) {
        if let CurrentState::Ongoing = self.current_state {
            let current_player_bits = match self.current_player {
                Player::A => self.player_a_bits,
                Player::B => self.player_b_bits,
            };
            
            if Self::is_win(current_player_bits) {
                self.current_state = CurrentState::Win(self.current_player);
                return;
            }

            if Self::is_board_full(self.mask) {
                self.current_state = CurrentState::Draw
            }
        }
    }

    fn has_four(bits: u64, shift: usize) -> bool {
        let pairs = bits & (bits >> shift);
        (pairs & (pairs >> (2 * shift))) != 0
    }

    pub fn is_win(bits: u64) -> bool {
        Self::has_four(bits, 1) // Vertical win check (adjacent bits)
        || Self::has_four(bits, Self::STRIDE) // Horizontal win check
        || Self::has_four(bits, Self::STRIDE - 1) // Diagonal win check (/)
        || Self::has_four(bits, Self::STRIDE + 1) // Diagonal win check (\)

    }

    /// Check for draw after checking that it isn't a win
    pub fn is_board_full(mask: u64) -> bool {
        (0..WIDTH).all(|col| !Self::can_play_helper(mask, col))
    }

    pub fn valid_actions(&self) -> Vec<usize> {
        (0..WIDTH).filter(|col| Self::can_play_helper(self.mask, *col)).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::connect4::{Connect4Env, CurrentState, Player};

    #[test]
    fn test_vertical_win() {
        let mut connect4: Connect4Env<6, 7> = Connect4Env::new();

        connect4.play_move(0); // A
        connect4.play_move(1); // B
        connect4.play_move(0); // A
        connect4.play_move(1); // B
        connect4.play_move(0); // A
        connect4.play_move(1); // B
        connect4.play_move(0); // A wins vertically

        assert!(connect4.current_state == CurrentState::Win(Player::A));
    }

    #[test]
    fn test_horizontal_win() {
        let mut connect4: Connect4Env<6, 7> = Connect4Env::new();

        connect4.play_move(0); // A
        connect4.play_move(4); // B
        connect4.play_move(1); // A
        connect4.play_move(4); // B
        connect4.play_move(2); // A
        connect4.play_move(5); // B
        connect4.play_move(3); // A wins horizontally

        assert!(connect4.current_state == CurrentState::Win(Player::A));
    }

    #[test]
    fn test_diagonal_backslash_win() {
        let mut connect4: Connect4Env<6, 7> = Connect4Env::new();

        // A target diagonal:
        //
        // row 3: . . . A
        // row 2: . . A B
        // row 1: . A B B
        // row 0: A B B B
        //
        // Coordinates for A: (0,0), (1,1), (2,2), (3,3)

        connect4.play_move(0); // A at col 0, row 0
        connect4.play_move(1); // B support

        connect4.play_move(1); // A at col 1, row 1
        connect4.play_move(2); // B support

        connect4.play_move(4); // A filler
        connect4.play_move(2); // B support

        connect4.play_move(2); // A at col 2, row 2
        connect4.play_move(3); // B support

        connect4.play_move(5); // A filler
        connect4.play_move(3); // B support

        connect4.play_move(5); // A filler
        connect4.play_move(3); // B support

        connect4.play_move(3); // A at col 3, row 3, wins diagonal \

        assert!(connect4.current_state == CurrentState::Win(Player::A));
    }

    #[test]
    fn test_diagonal_slash_win() {
        let mut connect4: Connect4Env<6, 7> = Connect4Env::new();

        // A target diagonal:
        //
        // row 3: A . . .
        // row 2: B A . .
        // row 1: B B A .
        // row 0: B B B A
        //
        // Coordinates for A: (3,0), (2,1), (1,2), (0,3)

        connect4.play_move(3); // A at col 3, row 0
        connect4.play_move(2); // B support

        connect4.play_move(2); // A at col 2, row 1
        connect4.play_move(1); // B support

        connect4.play_move(4); // A filler
        connect4.play_move(1); // B support

        connect4.play_move(1); // A at col 1, row 2
        connect4.play_move(0); // B support

        connect4.play_move(5); // A filler
        connect4.play_move(0); // B support

        connect4.play_move(5); // A filler
        connect4.play_move(0); // B support

        connect4.play_move(0); // A at col 0, row 3, wins diagonal /

        assert!(connect4.current_state == CurrentState::Win(Player::A));
    }

    #[test]
    fn test_valid_actions_removes_full_column() {
        let mut connect4: Connect4Env<6, 7> = Connect4Env::new();

        for _ in 0..7 {
            connect4.play_move(0);
        }

        assert!(!connect4.can_play(0));
        assert!(!connect4.valid_actions().contains(&0));
    }

    #[test]
    #[should_panic(expected = "Cannot play move: game is already over")]
    fn test_cannot_play_after_game_over() {
        let mut connect4: Connect4Env<6, 7> = Connect4Env::new();

        connect4.play_move(0); // A
        connect4.play_move(1); // B
        connect4.play_move(0); // A
        connect4.play_move(1); // B
        connect4.play_move(0); // A
        connect4.play_move(1); // B
        connect4.play_move(0); // A wins

        connect4.play_move(2); // should panic
    }
}
