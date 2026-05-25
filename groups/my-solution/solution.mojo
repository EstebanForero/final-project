from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort

from std.bit import pop_count

comptime WIDTH: UInt64  = 7
comptime STRIDE: UInt64  = WIDTH + 1
comptime HEIGHT: UInt64 = 6

# In select best move
comptime DEPTH: Int = 4

# Used for negamax win or lose situations
comptime BIG_SCORE: Int = 1000000

# Negamax to define order, first we try statistically stronger columns, from center to extremes
comptime COL_ORDER: InlineArray[Int, Int(WIDTH)] = [3, 2, 4, 1, 5, 0, 6]

def full_board_mask() -> UInt64:
    var mask: UInt64 = 0
    for c in range(Int(WIDTH)):
        mask |= ((UInt64(1) << HEIGHT) - 1) << (UInt64(c) * STRIDE)
    return mask

comptime FULL_BOARD: UInt64 = full_board_mask();

@export
def PyInit_solution() -> PythonObject:
    try:
        var m = PythonModuleBuilder("solution")
        m.def_function[act]("act", docstring="Pick a column to play")
        return m.finalize()
    except e:
        abort(String("error creating solution module: ", e))


def act(flat_board: PythonObject, depth_obj: PythonObject) raises -> PythonObject:
    # flat_board: float32 array of shape (HEIGHT*WIDTH,), row-major
    # current player's pieces = +1, opponent's pieces = -1, empty = 0
    # depth_obj: search depth (int)
    # return: column index to play (0–6)

    # TODO: implement solution start in here

    var board = from_flat_board(flat_board)
    return select_best_move(board)


def select_best_move(mut board: Bitboard) -> Int:
    var action_max = -1
    var score_max = -Int.MAX

    for col in COL_ORDER:

        if board.is_valid_move(col):
            board.make_move(col)
            var score = -negamax(board, DEPTH - 1, -Int.MAX, Int.MAX)
            board.undo_move(col)

            if score > score_max:
                score_max = score
                action_max = col

    return action_max


struct Bitboard(Movable):
    var my_pieces: UInt64
    var opp_pieces: UInt64
    var heights: InlineArray[UInt8, Int(WIDTH)]

    def __init__(out self):
        self.my_pieces = 0
        self.opp_pieces = 0
        self.heights = InlineArray[UInt8, Int(WIDTH)](fill=0)

    def is_valid_move(self, col: Int) -> Bool:
        return self.heights[col] < UInt8(HEIGHT)

    def make_move(mut self, col: Int):
        var bit = UInt64(1) << (STRIDE * UInt64(col) + UInt64(self.heights[col]))
        self.my_pieces |= bit
        self.heights[col] += 1
        (self.my_pieces, self.opp_pieces) = (self.opp_pieces, self.my_pieces)

    def undo_move(mut self, col: Int):
        (self.my_pieces, self.opp_pieces) = (self.opp_pieces, self.my_pieces)
        self.heights[col] -= 1
        var bit = UInt64(1) << (STRIDE * UInt64(col) + UInt64(self.heights[col]))
        self.my_pieces &= ~bit

    def check_win(self) -> Bool:
        return (
                self._wins_in(1, self.my_pieces) or # Vertical
                self._wins_in(STRIDE, self.my_pieces) or # Horizontal
                self._wins_in(STRIDE - 1, self.my_pieces) or # Diagonal (\)
                self._wins_in(STRIDE + 1, self.my_pieces) # Diagonal (/) 
                )

    def check_win_opp(self) -> Bool:
        return (
                self._wins_in(1, self.opp_pieces) or # Vertical
                self._wins_in(STRIDE, self.opp_pieces) or # Horizontal
                self._wins_in(STRIDE - 1, self.opp_pieces) or # Diagonal (\)
                self._wins_in(STRIDE + 1, self.opp_pieces) # Diagonal (/) 
                )


    def _wins_in(self, shift: UInt64, pieces: UInt64) -> Bool:
        var pairs = pieces & (pieces >> shift)
        return (pairs & (pairs >> (2 * shift))) != 0

    def is_draw(self) -> Bool:
        return (self.my_pieces | self.opp_pieces) == FULL_BOARD

    def piece_at(self, row: Int, col: Int) -> Int:
        var bit = UInt64(1) << (UInt64(col) * STRIDE + UInt64(row))

        if self.my_pieces & bit:
            return 1
        elif self.opp_pieces & bit:
            return -1
        return 0


def from_flat_board(flat_board: PythonObject) raises -> Bitboard:
    var board = Bitboard()

    for row in range(HEIGHT):
        for col in range(WIDTH):
            var idk = row * WIDTH + col
            var value = Float32(py=flat_board[idk]) # It can be -1 , +1 or 0
            var bit: UInt64 = 1 << UInt64(col * STRIDE + row)


            if value > 0.5:
                board.my_pieces |= bit
                board.heights[col] += 1
            elif value < -0.5:
                board.opp_pieces |= bit
                board.heights[col] += 1

    return board^


def negamax(mut board: Bitboard, depth: Int, alpha: Int, beta: Int) -> Int:
    if board.check_win_opp():
        return -BIG_SCORE
    elif board.is_draw():
        return 0
    elif depth == 0:
        return heuristic(board) # TODO: heuristics

    var best_score = -BIG_SCORE

    var local_alpha = alpha

    for col in COL_ORDER:
        if board.is_valid_move(col):
            board.make_move(col)
            var score = -negamax(board, depth - 1, -beta, -local_alpha)
            board.undo_move(col)

            if score > best_score:
                best_score = score
            if score > local_alpha:
                local_alpha = score
            if local_alpha >= beta:
                break

    return best_score

def window_score(window_me: Int, window_opp: Int) -> Int:
    if window_me > 0 and window_opp > 0:
        return 0 # Nobody can wi in there since the opponent already has pieces
    elif window_me == 3:
        return 5 # strong thread, one move from winning
    elif window_me == 2:
        return 2 # building thread for the enemy
    elif window_opp == 3:
        return -4 # opponent one move from winning
    elif window_opp == 2:
        return -1 # opponent building winning move
    return 0

def eval_window(board: Bitboard, row: Int, col: Int, delta_row: Int, delta_col: Int) -> Int:
    var my_count = 0
    var opp_count = 0

    for i in range(4):
        var v = board.piece_at(row + i * delta_row, col + i * delta_col)
        if v == 1:
            my_count += 1
        elif v == -1:
            opp_count += 1
    return window_score(my_count, opp_count)

def heuristic(board: Bitboard) -> Int:
    var score = 0

    # Horizontal check
    for row in range(0, HEIGHT):
        for col in range(0, WIDTH - 3):
            score += eval_window(board, row, col, delta_row = 0, delta_col = 1)

    # Vertical check
    for row in range(0, HEIGHT - 3):
        for col in range(0, WIDTH):
            score += eval_window(board, row, col, delta_row = 1, delta_col = 0)

    # Diagonal check (/)
    for row in range(0, HEIGHT - 3):
        for col in range(0, WIDTH - 3):
            score += eval_window(board, row, col, delta_row = 1, delta_col = 1)

    # Diagonal check (\)
    for row in range(3, HEIGHT):
        for col in range(0, WIDTH - 3):
            score += eval_window(board, row, col, delta_row = -1, delta_col = 1)


    return score
