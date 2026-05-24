from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort

comptime WIDTH: UInt64  = 7
comptime STRIDE: UInt64  = WIDTH + 1
comptime HEIGHT: UInt64 = 6


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

    # TODO: implement your solution here
    return 0

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




