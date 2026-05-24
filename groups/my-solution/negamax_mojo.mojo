from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.collections import List

# =============================================================================
# Constants
# =============================================================================

comptime WIDTH  = 7
comptime HEIGHT = 6
comptime STRIDE = HEIGHT + 1   # 7 — sentinel bit per column

comptime WIN_SCORE: Int32 = 100_000
comptime NEG_INF:  Int32  = -200_000
comptime POS_INF:  Int32  =  200_000

comptime TT_SIZE: Int = 1 << 18   # 262 144 entries ≈ 4 MB


# =============================================================================
# Module entry point
# =============================================================================

@export
def PyInit_negamax_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("negamax_mojo")
        m.def_function[act_plain]("act_plain", docstring="Negamax + alpha-beta, no TT")
        m.def_function[act_tt]("act_tt",       docstring="Negamax + alpha-beta + transposition table")
        return m.finalize()
    except e:
        abort(String("error creating negamax_mojo module: ", e))


def act_plain(flat_board: PythonObject, depth_obj: PythonObject) raises -> PythonObject:
    var board = BinaryBoard.from_flat_array(flat_board)
    if board.valid_mask == 0: return 0
    var depth = max(0, Int(py=depth_obj))
    return _choose_plain(board, depth)


def act_tt(flat_board: PythonObject, depth_obj: PythonObject) raises -> PythonObject:
    var board = BinaryBoard.from_flat_array(flat_board)
    if board.valid_mask == 0: return 0
    var depth = max(0, Int(py=depth_obj))
    var tt = TranspositionTable()
    return _choose_tt(board, depth, tt)


# =============================================================================
# Board
# =============================================================================

struct BinaryBoard(ImplicitlyCopyable):
    var cur:        UInt64   # current player's pieces
    var opp:        UInt64   # opponent's pieces
    var mask:       UInt64   # all occupied cells
    var valid_mask: Int      # bitmask: bit k = 1 if column k is playable

    def __init__(out self, cur: UInt64, opp: UInt64, mask: UInt64, vm: Int):
        self.cur        = cur
        self.opp        = opp
        self.mask       = mask
        self.valid_mask = vm

    @staticmethod
    def from_flat_array(flat: PythonObject) raises -> BinaryBoard:
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
        var move_bit = (self.mask + _bottom_mask(col)) & _column_mask(col)
        var new_mask = self.mask | move_bit
        return BinaryBoard(self.opp, self.cur | move_bit, new_mask, _valid_mask(new_mask))

    fn piece_at(self, row: Int, col: Int) -> Int32:
        var bit = UInt64(1) << UInt64(col * STRIDE + (HEIGHT - 1 - row))
        if (self.cur & bit) != UInt64(0): return Int32(1)
        if (self.opp & bit) != UInt64(0): return Int32(-1)
        return Int32(0)

    fn key(self) -> UInt64:
        var h = self.cur ^ (self.opp * UInt64(0x9E3779B97F4A7C15))
        h ^= h >> 33
        h *= UInt64(0xFF51AFD7ED558CCD)
        h ^= h >> 33
        return h


fn _bottom_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE)

fn _top_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE + HEIGHT - 1)

fn _column_mask(col: Int) -> UInt64:
    return ((UInt64(1) << UInt64(HEIGHT)) - UInt64(1)) << UInt64(col * STRIDE)

fn _valid_mask(mask: UInt64) -> Int:
    var vm = 0
    for col in range(WIDTH):
        if (mask & _top_mask(col)) == UInt64(0): vm |= 1 << col
    return vm

fn has_won(bits: UInt64) -> Bool:
    var m = bits & (bits >> UInt64(1))
    if (m & (m >> UInt64(2))) != UInt64(0): return True
    m = bits & (bits >> UInt64(STRIDE))
    if (m & (m >> UInt64(2 * STRIDE))) != UInt64(0): return True
    m = bits & (bits >> UInt64(HEIGHT))
    if (m & (m >> UInt64(2 * HEIGHT))) != UInt64(0): return True
    m = bits & (bits >> UInt64(HEIGHT + 2))
    if (m & (m >> UInt64(2 * (HEIGHT + 2)))) != UInt64(0): return True
    return False

fn _move_order(i: Int) -> Int:
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
# Heuristic  (same window scoring as Python Group B)
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
    var s: Int32 = 0
    for r in range(HEIGHT):
        for c in range(WIDTH - 3):
            s += _eval_window(board, r, c, 0, 1)
    for r in range(HEIGHT - 3):
        for c in range(WIDTH):
            s += _eval_window(board, r, c, 1, 0)
    for r in range(HEIGHT - 3):
        for c in range(WIDTH - 3):
            s += _eval_window(board, r, c, 1, 1)
    for r in range(HEIGHT - 3):
        for c in range(3, WIDTH):
            s += _eval_window(board, r, c, 1, -1)
    return s


# =============================================================================
# Plain negamax + alpha-beta
# =============================================================================

fn _negamax_plain(board: BinaryBoard, depth: Int, alpha_in: Int32, beta: Int32) -> Int32:
    if has_won(board.opp):    return -(WIN_SCORE + depth)
    if board.valid_mask == 0: return Int32(0)
    if depth <= 0:            return _heuristic(board)

    var alpha = alpha_in
    for i in range(WIDTH):
        var col = _move_order(i)
        if (board.valid_mask & (1 << col)) == 0: continue
        var child = board.play(col)
        var score = -_negamax_plain(child, depth - 1, -beta, -alpha)
        if score > alpha: alpha = score
        if alpha >= beta: break
    return alpha

fn _choose_plain(board: BinaryBoard, depth: Int) -> Int:
    var best_col   = _first_valid(board.valid_mask)
    var best_score = NEG_INF
    var alpha      = NEG_INF

    for i in range(WIDTH):
        var col = _move_order(i)
        if (board.valid_mask & (1 << col)) == 0: continue
        var child = board.play(col)
        if has_won(child.opp): return col          # immediate win
        var score = -_negamax_plain(child, depth - 1, -POS_INF, -alpha)
        if score > best_score:
            best_score = score
            best_col   = col
        if score > alpha: alpha = score
    return best_col


# =============================================================================
# Transposition table
# =============================================================================

struct TTEntry(ImplicitlyCopyable):
    var key:   UInt64
    var score: Int32
    var depth: Int8
    var flag:  UInt8   # 0=exact  1=lower-bound  2=upper-bound

    def __init__(out self):
        self.key   = UInt64(0)
        self.score = Int32(0)
        self.depth = Int8(-1)
        self.flag  = UInt8(0)

    def __init__(out self, key: UInt64, score: Int32, depth: Int8, flag: UInt8):
        self.key   = key
        self.score = score
        self.depth = depth
        self.flag  = flag


struct TranspositionTable(Movable):
    var entries: List[TTEntry]

    def __init__(out self):
        self.entries = List[TTEntry](capacity=TT_SIZE)
        for _ in range(TT_SIZE):
            self.entries.append(TTEntry())


# =============================================================================
# Negamax + alpha-beta + transposition table
# =============================================================================

fn _negamax_tt(
    board:    BinaryBoard,
    depth:    Int,
    alpha_in: Int32,
    beta_in:  Int32,
    mut tt:   TranspositionTable,
) -> Int32:
    if has_won(board.opp):    return -(WIN_SCORE + depth)
    if board.valid_mask == 0: return Int32(0)

    var k     = board.key()
    var idx   = Int(k & UInt64(TT_SIZE - 1))
    var alpha = alpha_in
    var beta  = beta_in

    # Probe TT
    var e = tt.entries[idx]
    if e.key == k and Int(e.depth) >= depth:
        if e.flag == UInt8(0): return e.score          # exact
        if e.flag == UInt8(1) and e.score > alpha: alpha = e.score   # lower bound
        elif e.flag == UInt8(2) and e.score < beta: beta = e.score   # upper bound
        if alpha >= beta: return e.score

    if depth <= 0:
        var h = _heuristic(board)
        tt.entries[idx] = TTEntry(k, h, Int8(0), UInt8(0))
        return h

    var orig_alpha = alpha
    var best       = NEG_INF

    for i in range(WIDTH):
        var col = _move_order(i)
        if (board.valid_mask & (1 << col)) == 0: continue
        var child = board.play(col)
        var score = -_negamax_tt(child, depth - 1, -beta, -alpha, tt)
        if score > best: best = score
        if score > alpha: alpha = score
        if alpha >= beta: break

    var flag = UInt8(0)
    if best <= orig_alpha: flag = UInt8(2)
    elif best >= beta:     flag = UInt8(1)
    tt.entries[idx] = TTEntry(k, best, Int8(depth), flag)
    return best

fn _choose_tt(board: BinaryBoard, depth: Int, mut tt: TranspositionTable) -> Int:
    var best_col   = _first_valid(board.valid_mask)
    var best_score = NEG_INF
    var alpha      = NEG_INF

    for i in range(WIDTH):
        var col = _move_order(i)
        if (board.valid_mask & (1 << col)) == 0: continue
        var child = board.play(col)
        if has_won(child.opp): return col
        var score = -_negamax_tt(child, depth - 1, -POS_INF, -alpha, tt)
        if score > best_score:
            best_score = score
            best_col   = col
        if score > alpha: alpha = score
    return best_col
