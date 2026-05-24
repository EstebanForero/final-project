from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.collections import List


# =============================================================================
# Architecture constants
# =============================================================================

comptime WIDTH  = 7
comptime HEIGHT = 6
comptime STRIDE = HEIGHT + 1

comptime INPUT_CHANNELS = 2
comptime CONV1_OUT      = 16
comptime CONV2_OUT      = 32
comptime HIDDEN         = 64
comptime BOARD_SIZE     = HEIGHT * WIDTH
comptime INPUT_SIZE     = INPUT_CHANNELS * BOARD_SIZE
comptime FLAT_SIZE      = CONV2_OUT * BOARD_SIZE

comptime CONV1_W_OFFSET = 0
comptime CONV1_W_SIZE   = CONV1_OUT * INPUT_CHANNELS * 3 * 3
comptime CONV1_B_OFFSET = CONV1_W_OFFSET + CONV1_W_SIZE
comptime CONV1_B_SIZE   = CONV1_OUT
comptime CONV2_W_OFFSET = CONV1_B_OFFSET + CONV1_B_SIZE
comptime CONV2_W_SIZE   = CONV2_OUT * CONV1_OUT * 3 * 3
comptime CONV2_B_OFFSET = CONV2_W_OFFSET + CONV2_W_SIZE
comptime CONV2_B_SIZE   = CONV2_OUT
comptime FC1_W_OFFSET   = CONV2_B_OFFSET + CONV2_B_SIZE
comptime FC1_W_SIZE     = HIDDEN * FLAT_SIZE
comptime FC1_B_OFFSET   = FC1_W_OFFSET + FC1_W_SIZE
comptime FC1_B_SIZE     = HIDDEN
comptime FC2_W_OFFSET   = FC1_B_OFFSET + FC1_B_SIZE
comptime FC2_W_SIZE     = WIDTH * HIDDEN
comptime FC2_B_OFFSET   = FC2_W_OFFSET + FC2_W_SIZE
comptime FC2_B_SIZE     = WIDTH
comptime TOTAL_WEIGHTS  = FC2_B_OFFSET + FC2_B_SIZE

# Search
comptime NEG_INF: Float32      = -999999.0
comptime NON_TERMINAL: Float32 =       2.0

# Hybrid evaluation blend: final = CNN_WEIGHT * cnn + (1-CNN_WEIGHT) * heuristic
# Heuristic is normalised via tanh_approx(raw / HEURISTIC_SCALE) → [-1, 1]
comptime CNN_WEIGHT: Float32      = 0.6
comptime HEURISTIC_SCALE: Float32 = 50.0

# Move ordering priority constants (large enough to stay above any heuristic score)
comptime WIN_PRIORITY:   Int = 1_000_000
comptime BLOCK_PRIORITY: Int =   500_000


# =============================================================================
# Module entry point
# =============================================================================

@export
def PyInit_act_hybrid_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_hybrid_mojo")
        m.def_function[act]("act", docstring="Hybrid CNN + heuristic negamax Connect 4 policy")
        return m.finalize()
    except e:
        abort(String("error creating act_hybrid_mojo module: ", e))


def act(
    flat_board:  PythonObject,
    rows_obj:    PythonObject,
    cols_obj:    PythonObject,
    timeout_obj: PythonObject,
    weights_obj: PythonObject,
    depth_obj:   PythonObject,
) raises -> PythonObject:
    var board = BinaryBoard.from_flat_array(flat_board)
    if board.valid_mask == 0:
        return 0

    var depth    = max(0, Int(py=depth_obj))
    var weights  = CnnWeights.from_numpy(weights_obj)
    var evaluator = CnnEvaluator(weights^)

    return HybridSearch.choose_action(board, evaluator, depth)


# =============================================================================
# Board representation
# =============================================================================

struct BinaryBoard(ImplicitlyCopyable):
    var current_player_bits: UInt64
    var opponent_bits:       UInt64
    var mask:                UInt64
    var valid_mask:          Int

    def __init__(
        out self,
        current_player_bits: UInt64,
        opponent_bits:       UInt64,
        mask:                UInt64,
        valid_mask:          Int,
    ):
        self.current_player_bits = current_player_bits
        self.opponent_bits       = opponent_bits
        self.mask                = mask
        self.valid_mask          = valid_mask

    @staticmethod
    def from_flat_array(flat_board: PythonObject) raises -> BinaryBoard:
        var cur  = UInt64(0)
        var opp  = UInt64(0)
        var mask = UInt64(0)
        for row in range(HEIGHT):
            for col in range(WIDTH):
                var v = Float32(py=flat_board.__getitem__(row * WIDTH + col))
                if v == 0.0: continue
                var bit = UInt64(1) << UInt64(col * STRIDE + (HEIGHT - 1 - row))
                mask |= bit
                if v > 0.0: cur |= bit
                else:       opp |= bit
        return BinaryBoard(cur, opp, mask, _valid_mask(mask))

    def terminal_value(self) -> Float32:
        if has_won(self.opponent_bits): return Float32(-1.0)
        if self.valid_mask == 0:        return Float32(0.0)
        return NON_TERMINAL

    def play(self, col: Int) -> BinaryBoard:
        var move_bit = (self.mask + _bottom_mask(col)) & _column_mask(col)
        var new_cur  = self.current_player_bits | move_bit
        var new_mask = self.mask | move_bit
        return BinaryBoard(self.opponent_bits, new_cur, new_mask, _valid_mask(new_mask))


fn _bottom_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE)

fn _top_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE + HEIGHT - 1)

fn _column_mask(col: Int) -> UInt64:
    return ((UInt64(1) << UInt64(HEIGHT)) - UInt64(1)) << UInt64(col * STRIDE)

fn _valid_mask(mask: UInt64) -> Int:
    var valid = 0
    for col in range(WIDTH):
        if (mask & _top_mask(col)) == UInt64(0):
            valid |= 1 << col
    return valid

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


# =============================================================================
# Heuristic evaluation
# =============================================================================

fn _cell_bit(row: Int, col: Int) -> UInt64:
    """Bit for board position (row from top, col)."""
    return UInt64(1) << UInt64(col * STRIDE + (HEIGHT - 1 - row))

fn _score_window(cur: UInt64, opp: UInt64, b0: UInt64, b1: UInt64, b2: UInt64, b3: UInt64) -> Int:
    """Score a 4-cell window from current player's perspective."""
    var n_cur = Int((cur & b0) != 0) + Int((cur & b1) != 0) + Int((cur & b2) != 0) + Int((cur & b3) != 0)
    var n_opp = Int((opp & b0) != 0) + Int((opp & b1) != 0) + Int((opp & b2) != 0) + Int((opp & b3) != 0)

    if n_cur > 0 and n_opp > 0: return 0  # dead window — mixed pieces

    if n_cur == 3: return 5   # one away from winning
    if n_cur == 2: return 2   # building threat
    if n_opp == 3: return -4  # opponent is one away from winning
    return 0

fn _heuristic(board: BinaryBoard) -> Int:
    """
    Window-based positional score from current player's perspective.
    Scans all 69 windows of 4 cells on the 6×7 board.
    """
    var cur   = board.current_player_bits
    var opp   = board.opponent_bits
    var score = 0

    # Horizontal
    for row in range(HEIGHT):
        for col in range(WIDTH - 3):
            score += _score_window(cur, opp,
                _cell_bit(row, col),   _cell_bit(row, col+1),
                _cell_bit(row, col+2), _cell_bit(row, col+3))

    # Vertical
    for row in range(HEIGHT - 3):
        for col in range(WIDTH):
            score += _score_window(cur, opp,
                _cell_bit(row,   col), _cell_bit(row+1, col),
                _cell_bit(row+2, col), _cell_bit(row+3, col))

    # Diagonal \ (top-left to bottom-right)
    for row in range(HEIGHT - 3):
        for col in range(WIDTH - 3):
            score += _score_window(cur, opp,
                _cell_bit(row,   col),   _cell_bit(row+1, col+1),
                _cell_bit(row+2, col+2), _cell_bit(row+3, col+3))

    # Diagonal / (bottom-left to top-right)
    for row in range(HEIGHT - 3):
        for col in range(3, WIDTH):
            score += _score_window(cur, opp,
                _cell_bit(row,   col),   _cell_bit(row+1, col-1),
                _cell_bit(row+2, col-2), _cell_bit(row+3, col-3))

    return score


# =============================================================================
# Heuristic-based move ordering
# =============================================================================

fn _move_priority(board: BinaryBoard, col: Int) -> Int:
    """
    Score a move for ordering purposes (higher = explore first).
    Cheap heuristic — no CNN call at internal nodes.
    """
    var child = board.play(col)

    # Immediate win — always explore first
    if has_won(child.opponent_bits):
        return WIN_PRIORITY

    # Blocking opponent's immediate win — explore second
    # Temporarily give opponent the move at this column
    var opp_child = BinaryBoard(
        board.opponent_bits,
        board.current_player_bits,
        board.mask,
        board.valid_mask,
    ).play(col)
    if has_won(opp_child.opponent_bits):
        return BLOCK_PRIORITY

    # Heuristic of resulting position (negated: good for me = bad for opponent)
    return -_heuristic(child)


struct OrderedMoves(ImplicitlyCopyable):
    """Legal moves sorted by heuristic priority, best first (selection sort on ≤7 items)."""
    var c0: Int; var c1: Int; var c2: Int; var c3: Int
    var c4: Int; var c5: Int; var c6: Int
    var count: Int

    def __init__(out self, board: BinaryBoard):
        self.c0=-1; self.c1=-1; self.c2=-1; self.c3=-1
        self.c4=-1; self.c5=-1; self.c6=-1
        self.count=0

        var used = 0
        for _ in range(WIDTH):
            var best_col   = -1
            var best_score = -999_999_999
            for col in range(WIDTH):
                if (board.valid_mask & (1 << col)) == 0: continue
                if (used          & (1 << col)) != 0: continue
                var s = _move_priority(board, col)
                if s > best_score:
                    best_score = s
                    best_col   = col
            if best_col < 0: break
            self._set(self.count, best_col)
            self.count += 1
            used |= (1 << best_col)

    fn get(self, i: Int) -> Int:
        if i==0: return self.c0
        if i==1: return self.c1
        if i==2: return self.c2
        if i==3: return self.c3
        if i==4: return self.c4
        if i==5: return self.c5
        return self.c6

    fn _set(mut self, i: Int, col: Int):
        if i==0: self.c0=col
        elif i==1: self.c1=col
        elif i==2: self.c2=col
        elif i==3: self.c3=col
        elif i==4: self.c4=col
        elif i==5: self.c5=col
        else: self.c6=col


# =============================================================================
# Model weights
# =============================================================================

struct CnnWeights(Movable):
    var values: List[Float32]

    def __init__(out self, var values: List[Float32]):
        self.values = values^

    @staticmethod
    def from_numpy(arr: PythonObject) raises -> CnnWeights:
        var n = Int(py=arr.__len__())
        if n != TOTAL_WEIGHTS:
            abort(String("weight count mismatch: expected ", TOTAL_WEIGHTS, ", got ", n))
        var values = List[Float32](capacity=TOTAL_WEIGHTS)
        for i in range(TOTAL_WEIGHTS):
            values.append(Float32(py=arr.__getitem__(i)))
        return CnnWeights(values^)

    fn get(self, i: Int) -> Float32: return self.values[i]

    fn conv1_w(self, oc: Int, ic: Int, kr: Int, kc: Int) -> Float32:
        return self.get(CONV1_W_OFFSET + ((oc * INPUT_CHANNELS + ic) * 3 + kr) * 3 + kc)
    fn conv1_b(self, oc: Int) -> Float32: return self.get(CONV1_B_OFFSET + oc)
    fn conv2_w(self, oc: Int, ic: Int, kr: Int, kc: Int) -> Float32:
        return self.get(CONV2_W_OFFSET + ((oc * CONV1_OUT + ic) * 3 + kr) * 3 + kc)
    fn conv2_b(self, oc: Int) -> Float32: return self.get(CONV2_B_OFFSET + oc)
    fn fc1_w(self, oi: Int, ii: Int) -> Float32: return self.get(FC1_W_OFFSET + oi * FLAT_SIZE + ii)
    fn fc1_b(self, oi: Int) -> Float32: return self.get(FC1_B_OFFSET + oi)
    fn fc2_w(self, oi: Int, ii: Int) -> Float32: return self.get(FC2_W_OFFSET + oi * HIDDEN + ii)
    fn fc2_b(self, oi: Int) -> Float32: return self.get(FC2_B_OFFSET + oi)


# =============================================================================
# CNN forward pass
# =============================================================================

struct QValues7(ImplicitlyCopyable):
    var q0: Float32; var q1: Float32; var q2: Float32; var q3: Float32
    var q4: Float32; var q5: Float32; var q6: Float32

    def __init__(out self):
        self.q0=0.0; self.q1=0.0; self.q2=0.0; self.q3=0.0
        self.q4=0.0; self.q5=0.0; self.q6=0.0

    fn get(self, col: Int) -> Float32:
        if col==0: return self.q0
        if col==1: return self.q1
        if col==2: return self.q2
        if col==3: return self.q3
        if col==4: return self.q4
        if col==5: return self.q5
        return self.q6

    fn set(mut self, col: Int, v: Float32):
        if col==0: self.q0=v
        elif col==1: self.q1=v
        elif col==2: self.q2=v
        elif col==3: self.q3=v
        elif col==4: self.q4=v
        elif col==5: self.q5=v
        else: self.q6=v

    fn best_among(self, valid_mask: Int) -> Float32:
        var best = NEG_INF
        for col in range(WIDTH):
            if (valid_mask & (1 << col)) == 0: continue
            var v = self.get(col)
            if v > best: best = v
        return best

    fn best_action(self, valid_mask: Int) -> Int:
        var best_col = _first_valid(valid_mask)
        var best_val = self.get(best_col)
        for col in range(WIDTH):
            if (valid_mask & (1 << col)) == 0: continue
            var v = self.get(col)
            if v > best_val:
                best_val = v
                best_col = col
        return best_col


fn _relu(x: Float32) -> Float32:
    return x if x > Float32(0.0) else Float32(0.0)

fn _tanh_approx(x: Float32) -> Float32:
    if x >= Float32(0.0): return x / (Float32(1.0) + x)
    return x / (Float32(1.0) - x)

fn _chw(ch: Int, row: Int, col: Int) -> Int:
    return ch * BOARD_SIZE + row * WIDTH + col


struct CnnEvaluator(Movable):
    var weights: CnnWeights

    def __init__(out self, var weights: CnnWeights):
        self.weights = weights^

    def predict(self, board: BinaryBoard) -> QValues7:
        var inp    = List[Float32](capacity=INPUT_SIZE)
        var h1     = List[Float32](capacity=CONV1_OUT * BOARD_SIZE)
        var h2     = List[Float32](capacity=CONV2_OUT * BOARD_SIZE)
        var hidden = List[Float32](capacity=HIDDEN)
        for _ in range(INPUT_SIZE):             inp.append(Float32(0.0))
        for _ in range(CONV1_OUT * BOARD_SIZE): h1.append(Float32(0.0))
        for _ in range(CONV2_OUT * BOARD_SIZE): h2.append(Float32(0.0))
        for _ in range(HIDDEN):                 hidden.append(Float32(0.0))

        self._encode(board, inp)
        self._conv1_relu(inp, h1)
        self._conv2_relu(h1, h2)
        self._fc1_relu(h2, hidden)
        return self._fc2_tanh(hidden)

    fn _encode(self, board: BinaryBoard, mut inp: List[Float32]):
        for col in range(WIDTH):
            for rfb in range(HEIGHT):
                var bit  = UInt64(1) << UInt64(col * STRIDE + rfb)
                var base = (HEIGHT - 1 - rfb) * WIDTH + col
                if (board.current_player_bits & bit) != UInt64(0): inp[base] = Float32(1.0)
                if (board.opponent_bits       & bit) != UInt64(0): inp[BOARD_SIZE + base] = Float32(1.0)

    fn _conv1_relu(self, inp: List[Float32], mut out: List[Float32]):
        for oc in range(CONV1_OUT):
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    var s = self.weights.conv1_b(oc)
                    for ic in range(INPUT_CHANNELS):
                        for kr in range(3):
                            var rr = row + kr - 1
                            if rr < 0 or rr >= HEIGHT: continue
                            for kc in range(3):
                                var cc = col + kc - 1
                                if cc < 0 or cc >= WIDTH: continue
                                s += inp[_chw(ic,rr,cc)] * self.weights.conv1_w(oc,ic,kr,kc)
                    out[_chw(oc,row,col)] = _relu(s)

    fn _conv2_relu(self, inp: List[Float32], mut out: List[Float32]):
        for oc in range(CONV2_OUT):
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    var s = self.weights.conv2_b(oc)
                    for ic in range(CONV1_OUT):
                        for kr in range(3):
                            var rr = row + kr - 1
                            if rr < 0 or rr >= HEIGHT: continue
                            for kc in range(3):
                                var cc = col + kc - 1
                                if cc < 0 or cc >= WIDTH: continue
                                s += inp[_chw(ic,rr,cc)] * self.weights.conv2_w(oc,ic,kr,kc)
                    out[_chw(oc,row,col)] = _relu(s)

    fn _fc1_relu(self, inp: List[Float32], mut out: List[Float32]):
        for oi in range(HIDDEN):
            var s = self.weights.fc1_b(oi)
            for ii in range(FLAT_SIZE): s += inp[ii] * self.weights.fc1_w(oi,ii)
            out[oi] = _relu(s)

    fn _fc2_tanh(self, hidden: List[Float32]) -> QValues7:
        var q = QValues7()
        for col in range(WIDTH):
            var s = self.weights.fc2_b(col)
            for i in range(HIDDEN): s += hidden[i] * self.weights.fc2_w(col,i)
            q.set(col, _tanh_approx(s))
        return q


# =============================================================================
# Hybrid negamax search
# =============================================================================

struct HybridSearch:
    """
    Negamax with two improvements over plain CNN negamax:

    1. Move ordering: heuristic-ranked (wins → blocks → positional score)
       at every internal node instead of fixed center-first order.
       Better alpha-beta pruning → more nodes pruned → faster or deeper search.

    2. Hybrid leaf evaluation:
       final = CNN_WEIGHT * cnn_best_q + (1-CNN_WEIGHT) * tanh(heuristic / SCALE)
       CNN covers long-range strategy; heuristic covers immediate tactical threats.
    """

    @staticmethod
    def choose_action(board: BinaryBoard, evaluator: CnnEvaluator, depth: Int) -> Int:
        if depth <= 0:
            return evaluator.predict(board).best_action(board.valid_mask)

        # Always take an immediate win first
        for i in range(WIDTH):
            var col = _move_order(i)
            if (board.valid_mask & (1 << col)) == 0: continue
            if has_won(board.play(col).opponent_bits):
                return col

        var moves      = OrderedMoves(board)
        var best_action = _first_valid(board.valid_mask)
        var best_score  = NEG_INF
        var alpha       = NEG_INF

        for i in range(moves.count):
            var col   = moves.get(i)
            var child = board.play(col)
            var score = -HybridSearch._negamax(child, evaluator, depth - 1, NEG_INF, -alpha)
            if score > best_score:
                best_score  = score
                best_action = col
            if score > alpha:
                alpha = score

        return best_action

    @staticmethod
    fn _negamax(
        board:    BinaryBoard,
        evaluator: CnnEvaluator,
        depth:    Int,
        alpha_in: Float32,
        beta:     Float32,
    ) -> Float32:
        var terminal = board.terminal_value()
        if terminal != NON_TERMINAL:
            return terminal

        if depth <= 0:
            return HybridSearch._leaf_eval(board, evaluator)

        var alpha = alpha_in
        var best  = NEG_INF
        var moves = OrderedMoves(board)

        for i in range(moves.count):
            var col   = moves.get(i)
            var child = board.play(col)
            var score: Float32
            if has_won(child.opponent_bits):
                score = Float32(1.0)
            else:
                score = -HybridSearch._negamax(child, evaluator, depth - 1, -beta, -alpha)

            if score > best:  best  = score
            if score > alpha: alpha = score
            if alpha >= beta: break

        return best

    @staticmethod
    fn _leaf_eval(board: BinaryBoard, evaluator: CnnEvaluator) -> Float32:
        """Blend CNN Q-value with heuristic positional score."""
        var cnn_score = evaluator.predict(board).best_among(board.valid_mask)
        var h_raw     = Float32(_heuristic(board)) / HEURISTIC_SCALE
        var h_score   = _tanh_approx(h_raw)
        return CNN_WEIGHT * cnn_score + (Float32(1.0) - CNN_WEIGHT) * h_score


# =============================================================================
# Search utilities
# =============================================================================

fn _move_order(i: Int) -> Int:
    if i==0: return 3
    if i==1: return 2
    if i==2: return 4
    if i==3: return 1
    if i==4: return 5
    if i==5: return 0
    return 6

fn _first_valid(valid_mask: Int) -> Int:
    for i in range(WIDTH):
        var col = _move_order(i)
        if (valid_mask & (1 << col)) != 0:
            return col
    return 0
