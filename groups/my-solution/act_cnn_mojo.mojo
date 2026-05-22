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

# Weight layout — contiguous offsets matching the .bin export order
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

# Search sentinels (Float32 range is more than sufficient)
comptime NEG_INF: Float32      = -999999.0
comptime NON_TERMINAL: Float32 =       2.0  # returned when the position is still live


# =============================================================================
# Module entry point
# =============================================================================

@export
def PyInit_act_cnn_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_cnn_mojo")
        m.def_function[act]("act", docstring="CNN + negamax Connect 4 policy")
        return m.finalize()
    except e:
        abort(String("error creating act_cnn_mojo module: ", e))


def act(
    flat_board:  PythonObject,   # float32 numpy array, shape (42,), already perspective-corrected
    rows_obj:    PythonObject,
    cols_obj:    PythonObject,
    timeout_obj: PythonObject,
    weights_obj: PythonObject,   # float32 numpy array, shape (TOTAL_WEIGHTS,), pre-loaded in Python
    depth_obj:   PythonObject,
) raises -> PythonObject:
    var board = BinaryBoard.from_flat_array(flat_board)
    if board.valid_mask == 0:
        return 0

    var depth = max(0, Int(py=depth_obj))
    var weights  = CnnWeights.from_numpy(weights_obj)
    var evaluator = CnnEvaluator(weights^)

    return NegamaxSearch.choose_action(board, evaluator, depth)


# =============================================================================
# Board representation
# =============================================================================

struct BinaryBoard(ImplicitlyCopyable):
    var current_player_bits: UInt64
    var opponent_bits:        UInt64
    var mask:                 UInt64
    var valid_mask:           Int

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
        """Decode a flat (42,) float32 array into the bitboard representation.
        Positive values = current player, negative = opponent."""
        var cur  = UInt64(0)
        var opp  = UInt64(0)
        var mask = UInt64(0)

        for row in range(HEIGHT):
            for col in range(WIDTH):
                var v = Float32(py=flat_board.__getitem__(row * WIDTH + col))
                if v == 0.0:
                    continue
                # Python row 0 is the top; bitboard row 0 is the bottom.
                var bit = UInt64(1) << UInt64(col * STRIDE + (HEIGHT - 1 - row))
                mask |= bit
                if v > 0.0:
                    cur |= bit
                else:
                    opp |= bit

        return BinaryBoard(cur, opp, mask, _valid_mask(mask))

    def is_terminal(self) -> Bool:
        return has_won(self.opponent_bits) or self.valid_mask == 0

    def terminal_value(self) -> Float32:
        """Score for the side to move. Returns NON_TERMINAL if the game is live."""
        if has_won(self.opponent_bits):
            return Float32(-1.0)  # the player who just moved won → current side lost
        if self.valid_mask == 0:
            return Float32(0.0)   # draw
        return NON_TERMINAL

    def play(self, col: Int) -> BinaryBoard:
        """Drop a piece in col and return the resulting board from the next player's view."""
        var move_bit    = (self.mask + _bottom_mask(col)) & _column_mask(col)
        var new_current = self.current_player_bits | move_bit
        var new_mask    = self.mask | move_bit
        return BinaryBoard(self.opponent_bits, new_current, new_mask, _valid_mask(new_mask))


# ── Bitboard helpers ─────────────────────────────────────────────────────────

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
    if (m & (m >> UInt64(2))) != UInt64(0): return True       # vertical
    m = bits & (bits >> UInt64(STRIDE))
    if (m & (m >> UInt64(2 * STRIDE))) != UInt64(0): return True  # horizontal
    m = bits & (bits >> UInt64(HEIGHT))
    if (m & (m >> UInt64(2 * HEIGHT))) != UInt64(0): return True   # diagonal /
    m = bits & (bits >> UInt64(HEIGHT + 2))
    if (m & (m >> UInt64(2 * (HEIGHT + 2)))) != UInt64(0): return True  # diagonal \
    return False


# =============================================================================
# Model weights  (Float32 — matches the exported .bin format)
# =============================================================================

struct CnnWeights(Movable):
    var values: List[Float32]

    def __init__(out self, var values: List[Float32]):
        self.values = values^

    @staticmethod
    def from_numpy(arr: PythonObject) raises -> CnnWeights:
        """Load weights from a pre-parsed numpy float32 array (avoids struct parsing per move)."""
        var n = Int(py=arr.__len__())
        if n != TOTAL_WEIGHTS:
            abort(String("weight count mismatch: expected ", TOTAL_WEIGHTS, ", got ", n))
        var values = List[Float32](capacity=TOTAL_WEIGHTS)
        for i in range(TOTAL_WEIGHTS):
            values.append(Float32(py=arr.__getitem__(i)))
        return CnnWeights(values^)

    fn get(self, i: Int) -> Float32:
        return self.values[i]

    # ── Named accessors for each layer ───────────────────────────────────────

    fn conv1_w(self, oc: Int, ic: Int, kr: Int, kc: Int) -> Float32:
        return self.get(CONV1_W_OFFSET + ((oc * INPUT_CHANNELS + ic) * 3 + kr) * 3 + kc)

    fn conv1_b(self, oc: Int) -> Float32:
        return self.get(CONV1_B_OFFSET + oc)

    fn conv2_w(self, oc: Int, ic: Int, kr: Int, kc: Int) -> Float32:
        return self.get(CONV2_W_OFFSET + ((oc * CONV1_OUT + ic) * 3 + kr) * 3 + kc)

    fn conv2_b(self, oc: Int) -> Float32:
        return self.get(CONV2_B_OFFSET + oc)

    fn fc1_w(self, oi: Int, ii: Int) -> Float32:
        return self.get(FC1_W_OFFSET + oi * FLAT_SIZE + ii)

    fn fc1_b(self, oi: Int) -> Float32:
        return self.get(FC1_B_OFFSET + oi)

    fn fc2_w(self, oi: Int, ii: Int) -> Float32:
        return self.get(FC2_W_OFFSET + oi * HIDDEN + ii)

    fn fc2_b(self, oi: Int) -> Float32:
        return self.get(FC2_B_OFFSET + oi)


# =============================================================================
# CNN forward pass
# =============================================================================

struct QValues7(ImplicitlyCopyable):
    """Q-value for each of the 7 columns, stored as individual fields for speed."""
    var q0: Float32; var q1: Float32; var q2: Float32; var q3: Float32
    var q4: Float32; var q5: Float32; var q6: Float32

    def __init__(out self):
        self.q0 = 0.0; self.q1 = 0.0; self.q2 = 0.0; self.q3 = 0.0
        self.q4 = 0.0; self.q5 = 0.0; self.q6 = 0.0

    fn get(self, col: Int) -> Float32:
        if col == 0: return self.q0
        if col == 1: return self.q1
        if col == 2: return self.q2
        if col == 3: return self.q3
        if col == 4: return self.q4
        if col == 5: return self.q5
        return self.q6

    fn set(mut self, col: Int, v: Float32):
        if col == 0: self.q0 = v
        elif col == 1: self.q1 = v
        elif col == 2: self.q2 = v
        elif col == 3: self.q3 = v
        elif col == 4: self.q4 = v
        elif col == 5: self.q5 = v
        else: self.q6 = v

    fn best_among(self, valid_mask: Int) -> Float32:
        var best = NEG_INF
        for i in range(WIDTH):
            var col = _move_order(i)
            if (valid_mask & (1 << col)) == 0:
                continue
            var v = self.get(col)
            if v > best:
                best = v
        return best

    fn best_action(self, valid_mask: Int) -> Int:
        var best_col = _first_valid(valid_mask)
        var best_val = self.get(best_col)
        for i in range(WIDTH):
            var col = _move_order(i)
            if (valid_mask & (1 << col)) == 0:
                continue
            var v = self.get(col)
            if v > best_val:
                best_val = v
                best_col = col
        return best_col


fn _relu(x: Float32) -> Float32:
    return x if x > Float32(0.0) else Float32(0.0)

fn _tanh_approx(x: Float32) -> Float32:
    """Monotonic squash into (-1, 1): x / (1 + |x|)."""
    if x >= Float32(0.0):
        return x / (Float32(1.0) + x)
    return x / (Float32(1.0) - x)

fn _chw(ch: Int, row: Int, col: Int) -> Int:
    return ch * BOARD_SIZE + row * WIDTH + col


struct CnnEvaluator(Movable):
    var weights: CnnWeights

    def __init__(out self, var weights: CnnWeights):
        self.weights = weights^

    def predict(self, board: BinaryBoard) -> QValues7:
        # Allocate activation buffers for this evaluation
        var inp    = List[Float32](capacity=INPUT_SIZE)
        var h1     = List[Float32](capacity=CONV1_OUT * BOARD_SIZE)
        var h2     = List[Float32](capacity=CONV2_OUT * BOARD_SIZE)
        var hidden = List[Float32](capacity=HIDDEN)
        for _ in range(INPUT_SIZE):             inp.append(Float32(0.0))
        for _ in range(CONV1_OUT * BOARD_SIZE): h1.append(Float32(0.0))
        for _ in range(CONV2_OUT * BOARD_SIZE): h2.append(Float32(0.0))
        for _ in range(HIDDEN):                 hidden.append(Float32(0.0))

        self._encode_board(board, inp)
        self._conv1_relu(inp, h1)
        self._conv2_relu(h1, h2)
        self._fc1_relu(h2, hidden)
        return self._fc2_tanh(hidden)

    fn _encode_board(self, board: BinaryBoard, mut inp: List[Float32]):
        for col in range(WIDTH):
            for rfb in range(HEIGHT):
                var bit  = UInt64(1) << UInt64(col * STRIDE + rfb)
                var base = (HEIGHT - 1 - rfb) * WIDTH + col
                if (board.current_player_bits & bit) != UInt64(0):
                    inp[base] = Float32(1.0)
                if (board.opponent_bits & bit) != UInt64(0):
                    inp[BOARD_SIZE + base] = Float32(1.0)

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
                                s += inp[_chw(ic, rr, cc)] * self.weights.conv1_w(oc, ic, kr, kc)
                    out[_chw(oc, row, col)] = _relu(s)

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
                                s += inp[_chw(ic, rr, cc)] * self.weights.conv2_w(oc, ic, kr, kc)
                    out[_chw(oc, row, col)] = _relu(s)

    fn _fc1_relu(self, inp: List[Float32], mut out: List[Float32]):
        for oi in range(HIDDEN):
            var s = self.weights.fc1_b(oi)
            for ii in range(FLAT_SIZE):
                s += inp[ii] * self.weights.fc1_w(oi, ii)
            out[oi] = _relu(s)

    fn _fc2_tanh(self, hidden: List[Float32]) -> QValues7:
        var q = QValues7()
        for col in range(WIDTH):
            var s = self.weights.fc2_b(col)
            for i in range(HIDDEN):
                s += hidden[i] * self.weights.fc2_w(col, i)
            q.set(col, _tanh_approx(s))
        return q


# =============================================================================
# Negamax search with alpha-beta pruning
# =============================================================================

struct NegamaxSearch:

    @staticmethod
    def choose_action(board: BinaryBoard, evaluator: CnnEvaluator, depth: Int) -> Int:
        """Return the best column to play from the current position."""
        if depth <= 0:
            return evaluator.predict(board).best_action(board.valid_mask)

        var best_action = _first_valid(board.valid_mask)
        var best_score  = NEG_INF
        var alpha       = NEG_INF

        for i in range(WIDTH):
            var col = _move_order(i)
            if (board.valid_mask & (1 << col)) == 0:
                continue

            var child = board.play(col)
            if has_won(child.opponent_bits):
                return col  # immediate win — take it

            var score = -NegamaxSearch._negamax(child, evaluator, depth - 1, NEG_INF, -alpha)
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
            return evaluator.predict(board).best_among(board.valid_mask)

        var alpha = alpha_in
        var best  = NEG_INF

        for i in range(WIDTH):
            var col = _move_order(i)
            if (board.valid_mask & (1 << col)) == 0:
                continue

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


# =============================================================================
# Search utilities
# =============================================================================

fn _move_order(i: Int) -> Int:
    """Center-first column ordering for better alpha-beta cut-offs."""
    if i == 0: return 3
    if i == 1: return 2
    if i == 2: return 4
    if i == 3: return 1
    if i == 4: return 5
    if i == 5: return 0
    return 6

fn _first_valid(valid_mask: Int) -> Int:
    for i in range(WIDTH):
        var col = _move_order(i)
        if (valid_mask & (1 << col)) != 0:
            return col
    return 0
