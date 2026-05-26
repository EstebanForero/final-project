from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort

from std.bit import pop_count

from std.python.bindings import PythonTypeBuilder

comptime WIDTH: UInt64  = 7
comptime HEIGHT: UInt64 = 6
comptime STRIDE: UInt64  = WIDTH + 1

# Used for negamax win or lose situations
comptime BIG_SCORE: Int32 = 1000000

# Negamax to define order, first we try statistically stronger columns, from center to extremes
comptime COL_ORDER: InlineArray[Int, Int(WIDTH)] = [3, 2, 4, 1, 5, 0, 6]

# Transposition table constants

comptime TT_SIZE = 1048576 # 2^20

def full_board_mask() -> UInt64:
    var mask: UInt64 = 0
    for c in range(Int(WIDTH)):
        mask |= ((UInt64(1) << HEIGHT) - 1) << (UInt64(c) * STRIDE)
    return mask

comptime FULL_BOARD: UInt64 = full_board_mask();


struct Agent(Defaultable, Movable, Writable):
    var tt: List[TTEntry]

    def __init__(out self):
        self.tt = List[TTEntry](capacity=TT_SIZE)
        for _ in range(TT_SIZE):
            self.tt.append(TTEntry())

    @staticmethod
    def act(self_ptr: UnsafePointer[Self, MutAnyOrigin], flat_board: PythonObject, depth_obj: PythonObject) raises -> PythonObject:
        #var self_ptr = py_self.downcast_value_ptr[Agent]()
        var board = from_flat_board(flat_board)
        var depth = Int8(max(0, Int(py=depth_obj)))
        return select_best_move(board, depth, self_ptr[].tt)

    def __str__(self) -> String:
        return String("Agent")

@export
def PyInit_solution() -> PythonObject:
    try:
        var m = PythonModuleBuilder("solution")
        _ = (
            m.add_type[Agent]("Agent")
            .def_init_defaultable[Agent]()
            .def_method[Agent.act]("act", docstring="Pick a column")
        )
        return m.finalize()
    except e:
        abort(String("error: ", e))

def select_best_move(board: Bitboard, depth: Int8, mut transposition_table: List[TTEntry]) -> Int:
    var action_max = -1
    var score_max = -Int32.MAX
    var alpha = -Int32.MAX

    for col in COL_ORDER:

        if board.is_valid_move(col):
            var child = board.play(col)
            var score = -negamax(child, depth - 1, -Int32.MAX, -alpha, transposition_table)

            if score > score_max:
                score_max = score
                action_max = col

            if score > alpha:
                alpha = score

    return action_max


struct Bitboard(Copyable):
    var my_pieces: UInt64
    var opp_pieces: UInt64
    var mask: UInt64

    def __init__(out self):
        self.my_pieces = 0
        self.opp_pieces = 0
        self.mask = 0

    def __init__(out self, my_pieces: UInt64, opp_pieces: UInt64, mask: UInt64):
        self.my_pieces  = my_pieces
        self.opp_pieces = opp_pieces
        self.mask       = mask

    def is_valid_move(self, col: Int) -> Bool:
        return (self.mask & _top_mask(col)) == 0

    def play(self, col: Int) -> Bitboard:
        var move_bit = (self.mask + _bottom_mask(col)) & _column_mask(col)
        var new_mask = self.mask | move_bit
        return Bitboard(self.opp_pieces, self.my_pieces | move_bit, new_mask)

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

    def key(self) -> UInt64: # Murmur hash finalizer
        var hash = self.my_pieces ^ (self.opp_pieces * UInt64(0x9E3779B97F4A7C15)) # Golden ratio constant
        hash ^= hash >> 33 # 64 bit integer gives the mest mixing, just over half
        hash *= UInt64(0xFF51AFD7ED558CCD) # MurmurHash3 paper constant?
        hash ^= hash >> 33
        return hash

fn _bottom_mask(col: Int) -> UInt64:
    return UInt64(1) << (UInt64(col) * STRIDE)

fn _top_mask(col: Int) -> UInt64:
    return UInt64(1) << (UInt64(col) * STRIDE + HEIGHT - 1)

fn _column_mask(col: Int) -> UInt64:
    return ((UInt64(1) << HEIGHT) - 1) << (UInt64(col) * STRIDE)


def from_flat_board(flat_board: PythonObject) raises -> Bitboard:
    var board = Bitboard()

    for row in range(HEIGHT):
        for col in range(WIDTH):
            var idk = row * WIDTH + col
            var value = Float32(py=flat_board[idk]) # It can be -1 , +1 or 0
            var bit: UInt64 = 1 << UInt64(col * STRIDE + row)


            if value > 0.5:
                board.my_pieces |= bit
                board.mask |= bit
            elif value < -0.5:
                board.opp_pieces |= bit
                board.mask |= bit

    return board^


def negamax(board: Bitboard, depth: Int8, alpha: Int32, beta: Int32, mut transposition_table: List[TTEntry]) -> Int32:
    if board.check_win_opp():
        return -(BIG_SCORE + Int32(depth))
    elif board.is_draw():
        return 0
    elif depth == 0:
        return heuristic(board)

    var best_score = -(BIG_SCORE + Int32(depth))

    var local_alpha = alpha
    var local_beta = beta

    # Index entry in tt
    var k = board.key()
    var idx = Int(k & UInt64(TT_SIZE - 1)) # Works only when the table size is a power of 2
    var entry = transposition_table[idx]

    if entry.hash == k and entry.depth >= depth:
        if entry.flag == 0: # exact score
            return entry.score
        elif entry.flag == 1: # lower bound tighten alpha
            if entry.score > local_alpha:
                local_alpha = entry.score
        elif entry.flag == 2: # upper bound tighten beta
            if entry.score < local_beta:
                local_beta = entry.score
        if local_alpha >= local_beta:
            return entry.score

    var orig_alpha = local_alpha

    for col in COL_ORDER:
        if board.is_valid_move(col):
            var child = board.play(col)
            var score = -negamax(child, depth - 1, -local_beta, -local_alpha, transposition_table)

            if score > best_score:
                best_score = score
            if score > local_alpha:
                local_alpha = score
            if local_alpha >= local_beta:
                break

    var flag: UInt8 = 0

    if best_score <= orig_alpha:
        flag = 2 # upper bound
    elif best_score >= local_beta:
        flag = 1 # lower bound
    transposition_table[idx] = TTEntry(k, best_score, depth, flag)

    return best_score

def window_score(window_me: Int32, window_opp: Int32) -> Int32:
    if window_me > 0 and window_opp > 0:
        return 0 # Nobody can wi in there since the opponent already has pieces
    elif window_me == 3:
        return 9 # strong thread, one move from winning
    elif window_me == 2:
        return 3 # building thread for the enemy
    elif window_opp == 3:
        return -10 # opponent one move from winning
    elif window_opp == 2:
        return 0 # we don't want to penalize just two in a row
    return 0

def eval_window(board: Bitboard, row: Int, col: Int, delta_row: Int, delta_col: Int) -> Int32:
    var my_count: Int32 = 0
    var opp_count: Int32 = 0

    for i in range(4):
        var v = board.piece_at(row + i * delta_row, col + i * delta_col)
        if v == 1:
            my_count += 1
        elif v == -1:
            opp_count += 1
    return window_score(my_count, opp_count)

def heuristic(board: Bitboard) -> Int32:
    var score: Int32 = 0

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

# Transposition table implementation

struct TTEntry(ImplicitlyCopyable, Movable, Writable):
    var hash: UInt64
    var score: Int32
    var depth: Int8
    var flag: UInt8

    def __init__(out self):
        self.hash = 0
        self.score = 0
        self.depth =0
        self.flag = 0

    def __init__(out self, key: UInt64, score: Int32, depth: Int8, flag: UInt8):
        self.hash = key
        self.score = score
        self.depth = depth
        self.flag  = flag
