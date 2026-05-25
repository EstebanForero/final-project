from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort

comptime WIDTH: UInt64  = 7
comptime STRIDE: UInt64  = WIDTH + 1
comptime HEIGHT: UInt64 = 6

# In select best move
comptime DEPTH: Int = 4

# Used for negamax win or lose situations
comptime BIG_SCORE: Int = 1000000

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
    var score_max = -BIG_SCORE

    for col in range(0, 7):

        if board.is_valid_move(col):
            board.make_move(col)
            var score = -negamax(board, DEPTH - 1, -BIG_SCORE, BIG_SCORE)
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
    if depth == 0:
        return 0 # TODO: heuristics

    var best_score = -BIG_SCORE

    var local_alpha = alpha

    for col in range(0, 7):
        if board.is_valid_move(col):
            board.make_move(col)
            var score = -negamax(board, depth - 1, -beta, -local_alpha)
            board.undo_move(col)

            if score > best_score:
                best_score = score
            if score > alpha:
                local_alpha = score
            if local_alpha >= beta:
                break

    return best_score
