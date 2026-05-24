from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.collections import List

# =============================================================================
# Board constants
# =============================================================================

comptime WIDTH  = 7
comptime HEIGHT = 6
comptime STRIDE = HEIGHT + 1   # sentinel bit between columns


# =============================================================================
# Module entry point
# =============================================================================

@export
def PyInit_solution() -> PythonObject:
    try:
        var m = PythonModuleBuilder("solution")
        m.def_function[act]("act", docstring="Pick a column to play")
        return m.finalize()
    except e:
        abort(String("error creating solution module: ", e))


def act(flat_board: PythonObject, depth_obj: PythonObject) raises -> PythonObject:
    var board = BinaryBoard.from_flat_array(flat_board)
    if board.valid_mask == 0: return 0
    var depth = max(0, Int(py=depth_obj))

    # -------------------------------------------------------------------------
    # TODO: implement your search here.
    # `board`  — current position (current player = cur, opponent = opp)
    # `depth`  — how many plies to search
    # Use board.play(col) to get the next position after playing column `col`.
    # Use has_won(bits) to check for a four-in-a-row.
    # Use _heuristic(board) for a position score when you hit depth 0.
    # -------------------------------------------------------------------------

    # Placeholder: play the first legal column in center-first order.
    return _first_valid(board.valid_mask)


# =============================================================================
# BinaryBoard  — bitboard representation
#
#   Each column is stored in 7 bits (HEIGHT+1=7, with a sentinel bit at the top).
#   Bit index for (row, col):  col * STRIDE + (HEIGHT - 1 - row)
#   Row 0 = top of board, row HEIGHT-1 = bottom.
#
#   cur  — bitmask of the current player's pieces
#   opp  — bitmask of the opponent's pieces
#   mask — bitmask of all occupied cells (cur | opp)
#   valid_mask — integer where bit k = 1 means column k can be played
# =============================================================================

struct BinaryBoard(ImplicitlyCopyable):
    var cur:        UInt64
    var opp:        UInt64
    var mask:       UInt64
    var valid_mask: Int

    def __init__(out self, cur: UInt64, opp: UInt64, mask: UInt64, vm: Int):
        self.cur        = cur
        self.opp        = opp
        self.mask       = mask
        self.valid_mask = vm

    @staticmethod
    def from_flat_array(flat: PythonObject) raises -> BinaryBoard:
        """Build a BinaryBoard from a flat float32 array (row-major, current player = +1)."""
        var cur  = UInt64(0); var opp = UInt64(0); var mask = UInt64(0)
        for row in range(HEIGHT):
            for col in range(WIDTH):
                var v = Float32(py=flat.__getitem__(row * WIDTH + col))
                if v == Float32(0.0): continue
                var bit = UInt64(1) << UInt64(col * STRIDE + (HEIGHT - 1 - row))
                mask |= bit
                if v > Float32(0.0): cur |= bit
                else:                opp |= bit
        return BinaryBoard(cur, opp, mask, _valid_mask(mask))

    def play(self, col: Int) -> BinaryBoard:
        """Return the board after the current player drops a piece in `col`.
        The returned board has cur/opp swapped (it is now the other player's turn)."""
        var move_bit = (self.mask + _bottom_mask(col)) & _column_mask(col)
        var new_mask = self.mask | move_bit
        return BinaryBoard(self.opp, self.cur | move_bit, new_mask, _valid_mask(new_mask))

    fn piece_at(self, row: Int, col: Int) -> Int32:
        """Returns +1 (current player), -1 (opponent), or 0 (empty)."""
        var bit = UInt64(1) << UInt64(col * STRIDE + (HEIGHT - 1 - row))
        if (self.cur & bit) != UInt64(0): return Int32(1)
        if (self.opp & bit) != UInt64(0): return Int32(-1)
        return Int32(0)


# =============================================================================
# Board utilities
# =============================================================================

fn _bottom_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE)

fn _top_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE + HEIGHT - 1)

fn _column_mask(col: Int) -> UInt64:
    return ((UInt64(1) << UInt64(HEIGHT)) - UInt64(1)) << UInt64(col * STRIDE)

fn _valid_mask(mask: UInt64) -> Int:
    """Return bitmask of playable columns (column not full)."""
    var vm = 0
    for col in range(WIDTH):
        if (mask & _top_mask(col)) == UInt64(0): vm |= 1 << col
    return vm

fn has_won(bits: UInt64) -> Bool:
    """True if the piece set `bits` contains four in a row."""
    var m = bits & (bits >> UInt64(1))
    if (m & (m >> UInt64(2))) != UInt64(0): return True        # horizontal
    m = bits & (bits >> UInt64(STRIDE))
    if (m & (m >> UInt64(2 * STRIDE))) != UInt64(0): return True  # vertical
    m = bits & (bits >> UInt64(HEIGHT))
    if (m & (m >> UInt64(2 * HEIGHT))) != UInt64(0): return True   # diagonal /
    m = bits & (bits >> UInt64(HEIGHT + 2))
    if (m & (m >> UInt64(2 * (HEIGHT + 2)))) != UInt64(0): return True  # diagonal \
    return False

fn _move_order(i: Int) -> Int:
    """Center-first column order: 3,2,4,1,5,0,6"""
    if i==0: return 3
    if i==1: return 2
    if i==2: return 4
    if i==3: return 1
    if i==4: return 5
    if i==5: return 0
    return 6

fn _first_valid(vm: Int) -> Int:
    for i in range(WIDTH):
        var col = _move_order(i)
        if (vm & (1 << col)) != 0: return col
    return 0


# =============================================================================
# Heuristic  — window-based position score from current player's perspective
#
# Score per 4-cell window:
#   both players present → 0   (blocked)
#   current player 4    → +100
#   current player 3    → +5
#   current player 2    → +2
#   opponent 3          → -4
# =============================================================================

fn _window_score(p: Int, o: Int) -> Int32:
    if p > 0 and o > 0: return Int32(0)
    if p == 4: return Int32(100)
    if p == 3: return Int32(5)
    if p == 2: return Int32(2)
    if o == 3: return Int32(-4)
    return Int32(0)

fn _eval_window(board: BinaryBoard, r0: Int, c0: Int, dr: Int, dc: Int) -> Int32:
    var p = 0; var o = 0
    for i in range(4):
        var v = board.piece_at(r0 + i*dr, c0 + i*dc)
        if v > Int32(0): p += 1
        elif v < Int32(0): o += 1
    return _window_score(p, o)

fn _heuristic(board: BinaryBoard) -> Int32:
    """Static evaluation from the current player's perspective."""
    var s: Int32 = 0
    for r in range(HEIGHT):
        for c in range(WIDTH - 3):
            s += _eval_window(board, r, c, 0, 1)   # horizontal
    for r in range(HEIGHT - 3):
        for c in range(WIDTH):
            s += _eval_window(board, r, c, 1, 0)   # vertical
    for r in range(HEIGHT - 3):
        for c in range(WIDTH - 3):
            s += _eval_window(board, r, c, 1, 1)   # diagonal down-right
    for r in range(HEIGHT - 3):
        for c in range(3, WIDTH):
            s += _eval_window(board, r, c, 1, -1)  # diagonal down-left
    return s
