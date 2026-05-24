from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.collections import List
from std.math import tanh


# =============================================================================
# Architecture constants — V-value model (1 output from fc2)
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
comptime FC2_W_SIZE     = 1 * HIDDEN   # V-model: single output
comptime FC2_B_OFFSET   = FC2_W_OFFSET + FC2_W_SIZE
comptime FC2_B_SIZE     = 1
comptime TOTAL_WEIGHTS  = FC2_B_OFFSET + FC2_B_SIZE   # 91,089

comptime NEG_INF: Float32      = -999999.0
comptime NON_TERMINAL: Float32 =       2.0


# =============================================================================
# Module entry point
# =============================================================================

@export
def PyInit_act_v_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_v_mojo")
        m.def_function[act]("act", docstring="V-value CNN + negamax Connect 4 policy")
        return m.finalize()
    except e:
        abort(String("error creating act_v_mojo module: ", e))


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

    return NegamaxSearch.choose_action(board, evaluator, depth)


# =============================================================================
# Board representation  (identical to act_cnn_mojo.mojo)
# =============================================================================

struct BinaryBoard(ImplicitlyCopyable):
    var current_player_bits: UInt64
    var opponent_bits:        UInt64
    var mask:                 UInt64
    var valid_mask:           Int

    def __init__(out self, cp: UInt64, opp: UInt64, mask: UInt64, vm: Int):
        self.current_player_bits = cp
        self.opponent_bits       = opp
        self.mask                = mask
        self.valid_mask          = vm

    @staticmethod
    def from_flat_array(flat_board: PythonObject) raises -> BinaryBoard:
        var cur  = UInt64(0); var opp = UInt64(0); var mask = UInt64(0)
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
        var new_mask = self.mask | move_bit
        return BinaryBoard(self.opponent_bits, self.current_player_bits | move_bit,
                           new_mask, _valid_mask(new_mask))


fn _bottom_mask(col: Int) -> UInt64: return UInt64(1) << UInt64(col * STRIDE)
fn _top_mask(col: Int) -> UInt64:    return UInt64(1) << UInt64(col * STRIDE + HEIGHT - 1)
fn _column_mask(col: Int) -> UInt64:
    return ((UInt64(1) << UInt64(HEIGHT)) - UInt64(1)) << UInt64(col * STRIDE)
fn _valid_mask(mask: UInt64) -> Int:
    var valid = 0
    for col in range(WIDTH):
        if (mask & _top_mask(col)) == UInt64(0): valid |= 1 << col
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
# Model weights  (V-model: fc2 has 1×64 weight + 1 bias)
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
    fn fc2_w(self, ii: Int) -> Float32: return self.get(FC2_W_OFFSET + ii)  # single output row
    fn fc2_b(self) -> Float32: return self.get(FC2_B_OFFSET)


# =============================================================================
# CNN forward pass — outputs single V-value
# =============================================================================

fn _relu(x: Float32) -> Float32: return x if x > Float32(0.0) else Float32(0.0)
fn _chw(ch: Int, row: Int, col: Int) -> Int: return ch * BOARD_SIZE + row * WIDTH + col


struct CnnEvaluator(Movable):
    var weights: CnnWeights

    def __init__(out self, var weights: CnnWeights):
        self.weights = weights^

    def predict_value(self, board: BinaryBoard) -> Float32:
        """Returns V ∈ (-1, 1): how good this position is for the current player."""
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
        return self._fc2_value(hidden)

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

    fn _fc2_value(self, hidden: List[Float32]) -> Float32:
        var v = self.weights.fc2_b()
        for i in range(HIDDEN): v += hidden[i] * self.weights.fc2_w(i)
        return tanh(v)


# =============================================================================
# Negamax search — V-value at leaves, center-first ordering at internal nodes
# =============================================================================

struct NegamaxSearch:

    @staticmethod
    def choose_action(board: BinaryBoard, evaluator: CnnEvaluator, depth: Int) -> Int:
        if depth <= 0:
            return NegamaxSearch._best_by_value(board, evaluator)

        var best_action = _first_valid(board.valid_mask)
        var best_score  = NEG_INF
        var alpha       = NEG_INF

        for i in range(WIDTH):
            var col = _move_order(i)
            if (board.valid_mask & (1 << col)) == 0: continue
            var child = board.play(col)
            if has_won(child.opponent_bits): return col  # immediate win

            var score = -NegamaxSearch._negamax(child, evaluator, depth - 1, NEG_INF, -alpha)
            if score > best_score:
                best_score  = score
                best_action = col
            if score > alpha: alpha = score

        return best_action

    @staticmethod
    fn _best_by_value(board: BinaryBoard, evaluator: CnnEvaluator) -> Int:
        """Greedy: pick the child with the highest V-value for us."""
        var best_action = _first_valid(board.valid_mask)
        var best_val    = NEG_INF
        for i in range(WIDTH):
            var col = _move_order(i)
            if (board.valid_mask & (1 << col)) == 0: continue
            var child = board.play(col)
            # child's V is from opponent's view → negate for our view
            var v = -evaluator.predict_value(child)
            if v > best_val:
                best_val    = v
                best_action = col
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
        if terminal != NON_TERMINAL: return terminal

        if depth <= 0:
            return evaluator.predict_value(board)  # direct V-value at leaf

        var alpha = alpha_in
        var best  = NEG_INF

        for i in range(WIDTH):
            var col = _move_order(i)
            if (board.valid_mask & (1 << col)) == 0: continue
            var child = board.play(col)
            var score: Float32
            if has_won(child.opponent_bits):
                score = Float32(1.0)
            else:
                score = -NegamaxSearch._negamax(child, evaluator, depth - 1, -beta, -alpha)

            if score > best:  best  = score
            if score > alpha: alpha = score
            if alpha >= beta: break

        return best


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
        if (valid_mask & (1 << col)) != 0: return col
    return 0
