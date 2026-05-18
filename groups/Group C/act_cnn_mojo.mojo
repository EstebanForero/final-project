from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.collections import List


comptime WIDTH = 7
comptime HEIGHT = 6
comptime STRIDE = HEIGHT + 1

comptime INPUT_CHANNELS = 2
comptime CONV1_OUT = 16
comptime CONV2_OUT = 32
comptime HIDDEN = 64
comptime BOARD_SIZE = HEIGHT * WIDTH
comptime INPUT_SIZE = INPUT_CHANNELS * BOARD_SIZE
comptime FLAT_SIZE = CONV2_OUT * BOARD_SIZE

comptime CONV1_W_OFFSET = 0
comptime CONV1_W_SIZE = CONV1_OUT * INPUT_CHANNELS * 3 * 3
comptime CONV1_B_OFFSET = CONV1_W_OFFSET + CONV1_W_SIZE
comptime CONV1_B_SIZE = CONV1_OUT

comptime CONV2_W_OFFSET = CONV1_B_OFFSET + CONV1_B_SIZE
comptime CONV2_W_SIZE = CONV2_OUT * CONV1_OUT * 3 * 3
comptime CONV2_B_OFFSET = CONV2_W_OFFSET + CONV2_W_SIZE
comptime CONV2_B_SIZE = CONV2_OUT

comptime FC1_W_OFFSET = CONV2_B_OFFSET + CONV2_B_SIZE
comptime FC1_W_SIZE = HIDDEN * FLAT_SIZE
comptime FC1_B_OFFSET = FC1_W_OFFSET + FC1_W_SIZE
comptime FC1_B_SIZE = HIDDEN

comptime FC2_W_OFFSET = FC1_B_OFFSET + FC1_B_SIZE
comptime FC2_W_SIZE = WIDTH * HIDDEN
comptime FC2_B_OFFSET = FC2_W_OFFSET + FC2_W_SIZE
comptime FC2_B_SIZE = WIDTH

comptime TOTAL_WEIGHTS = FC2_B_OFFSET + FC2_B_SIZE
comptime MODEL_BYTES = TOTAL_WEIGHTS * 4


@export
def PyInit_act_cnn_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_cnn_mojo")
        m.def_function[act]("act", docstring="CNN + negamax Connect 4 policy")
        return m.finalize()
    except e:
        abort(String("error creating act_cnn_mojo module: ", e))


def act(
    flat_board: PythonObject,
    rows_obj: PythonObject,
    cols_obj: PythonObject,
    timeout_obj: PythonObject,
    model_path_obj: PythonObject,
    depth_obj: PythonObject,
) raises -> PythonObject:
    var board = BinaryBoard.from_python_board(flat_board)

    if board.valid_mask == 0:
        return 0

    var depth = Int(py=depth_obj)
    if depth < 0:
        depth = 0

    var weights = CnnWeights.from_file(model_path_obj)
    var evaluator = CnnEvaluator(weights^)

    return NegamaxSearch.choose_action(board, evaluator, depth)


# -----------------------------------------------------------------------------
# Python-side cached raw model bytes
# -----------------------------------------------------------------------------

def get_model_cache() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var cache_name = "_act_cnn_mojo_raw_model_cache"
    var builtins_dict = builtins.__dict__

    if Bool(py=builtins_dict.__contains__(cache_name)):
        return builtins_dict.__getitem__(cache_name)

    var cache = builtins.dict()
    builtins_dict.__setitem__(cache_name, value=cache)
    return cache


def get_cached_model_bytes(path_obj: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var path_key = builtins.str(path_obj)
    var cache = get_model_cache()

    if Bool(py=cache.__contains__(path_key)):
        return cache.__getitem__(path_key)

    var file = builtins.open(path_obj, "rb")
    var data = file.read()
    file.close()

    cache.__setitem__(path_key, value=data)
    return data


# -----------------------------------------------------------------------------
# Board representation
# -----------------------------------------------------------------------------

struct BinaryBoard:
    var current_player_bits: UInt64
    var opponent_bits: UInt64
    var mask: UInt64
    var valid_mask: Int

    def __init__(
        out self,
        current_player_bits: UInt64,
        opponent_bits: UInt64,
        mask: UInt64,
        valid_mask: Int,
    ):
        self.current_player_bits = current_player_bits
        self.opponent_bits = opponent_bits
        self.mask = mask
        self.valid_mask = valid_mask

    @staticmethod
    def from_python_board(flat_board: PythonObject) raises -> BinaryBoard:
        var current_player_bits = UInt64(0)
        var opponent_bits = UInt64(0)
        var mask = UInt64(0)

        for row in range(HEIGHT):
            for col in range(WIDTH):
                var flat_index = row * WIDTH + col
                var value = Float64(py=flat_board.__getitem__(flat_index))

                if value == 0.0:
                    continue

                # Python board row 0 is top; bitboard row 0 is bottom.
                var bit_row = HEIGHT - 1 - row
                var bit_index = col * STRIDE + bit_row
                var bit = UInt64(1) << UInt64(bit_index)

                mask |= bit

                if value > 0.0:
                    current_player_bits |= bit
                else:
                    opponent_bits |= bit

        var valid_mask = compute_valid_mask(mask)

        return BinaryBoard(
            current_player_bits,
            opponent_bits,
            mask,
            valid_mask,
        )

    def is_draw(self) -> Bool:
        return self.valid_mask == 0

    def terminal_value_for_current_player(self) -> Float64:
        # Because play() swaps perspective, if opponent_bits has a four-in-a-row,
        # the previous mover won and the side to move has lost.
        if has_won(self.opponent_bits):
            return -1.0

        if self.is_draw():
            return 0.0

        # Non-terminal marker.
        return 2.0

    def play(self, action: Int) -> BinaryBoard:
        var move_bit = (self.mask + bottom_mask(action)) & column_mask(action)
        var new_current_bits = self.current_player_bits | move_bit
        var new_mask = self.mask | move_bit
        var new_valid_mask = compute_valid_mask(new_mask)

        # Swap perspective: next state is from the next player's view.
        return BinaryBoard(
            self.opponent_bits,
            new_current_bits,
            new_mask,
            new_valid_mask,
        )


def bottom_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE)


def top_mask(col: Int) -> UInt64:
    return UInt64(1) << UInt64(col * STRIDE + HEIGHT - 1)


def column_mask(col: Int) -> UInt64:
    return ((UInt64(1) << UInt64(HEIGHT)) - UInt64(1)) << UInt64(col * STRIDE)


def can_play(mask: UInt64, col: Int) -> Bool:
    return (mask & top_mask(col)) == UInt64(0)


def compute_valid_mask(mask: UInt64) -> Int:
    var valid = 0

    for col in range(WIDTH):
        if can_play(mask, col):
            valid |= 1 << col

    return valid


def has_won(bits: UInt64) -> Bool:
    # vertical
    var m = bits & (bits >> UInt64(1))
    if (m & (m >> UInt64(2))) != UInt64(0):
        return True

    # horizontal
    m = bits & (bits >> UInt64(STRIDE))
    if (m & (m >> UInt64(2 * STRIDE))) != UInt64(0):
        return True

    # diagonal /
    m = bits & (bits >> UInt64(HEIGHT))
    if (m & (m >> UInt64(2 * HEIGHT))) != UInt64(0):
        return True

    # diagonal \
    m = bits & (bits >> UInt64(HEIGHT + 2))
    if (m & (m >> UInt64(2 * (HEIGHT + 2)))) != UInt64(0):
        return True

    return False


# -----------------------------------------------------------------------------
# Model weights
# -----------------------------------------------------------------------------

struct CnnWeights:
    var values: List[Float64]

    def __init__(out self, var values: List[Float64]):
        self.values = values^

    @staticmethod
    def from_file(path_obj: PythonObject) raises -> CnnWeights:
        var data = get_cached_model_bytes(path_obj)
        var length = Int(py=data.__len__())

        if length != MODEL_BYTES:
            abort(
                String(
                    "bad model size: expected ",
                    MODEL_BYTES,
                    " bytes, got ",
                    length,
                )
            )

        var struct_module = Python.import_module("struct")
        var values = List[Float64]()

        var offset = 0

        for _ in range(TOTAL_WEIGHTS):
            var tuple_value = struct_module.unpack_from("<f", data, offset)
            values.append(Float64(py=tuple_value.__getitem__(0)))
            offset += 4

        return CnnWeights(values^)

    def get(self, index: Int) -> Float64:
        return self.values[index]

    def conv1_w(self, out_ch: Int, in_ch: Int, kr: Int, kc: Int) -> Float64:
        var idx = CONV1_W_OFFSET + (((out_ch * INPUT_CHANNELS + in_ch) * 3 + kr) * 3 + kc)
        return self.get(idx)

    def conv1_b(self, out_ch: Int) -> Float64:
        return self.get(CONV1_B_OFFSET + out_ch)

    def conv2_w(self, out_ch: Int, in_ch: Int, kr: Int, kc: Int) -> Float64:
        var idx = CONV2_W_OFFSET + (((out_ch * CONV1_OUT + in_ch) * 3 + kr) * 3 + kc)
        return self.get(idx)

    def conv2_b(self, out_ch: Int) -> Float64:
        return self.get(CONV2_B_OFFSET + out_ch)

    def fc1_w(self, out_i: Int, in_i: Int) -> Float64:
        return self.get(FC1_W_OFFSET + out_i * FLAT_SIZE + in_i)

    def fc1_b(self, out_i: Int) -> Float64:
        return self.get(FC1_B_OFFSET + out_i)

    def fc2_w(self, out_i: Int, in_i: Int) -> Float64:
        return self.get(FC2_W_OFFSET + out_i * HIDDEN + in_i)

    def fc2_b(self, out_i: Int) -> Float64:
        return self.get(FC2_B_OFFSET + out_i)


# -----------------------------------------------------------------------------
# Small fixed-size vector containers
# -----------------------------------------------------------------------------

struct QValues7:
    var q0: Float64
    var q1: Float64
    var q2: Float64
    var q3: Float64
    var q4: Float64
    var q5: Float64
    var q6: Float64

    def __init__(out self):
        self.q0 = 0.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.q4 = 0.0
        self.q5 = 0.0
        self.q6 = 0.0

    def get(self, action: Int) -> Float64:
        if action == 0:
            return self.q0
        elif action == 1:
            return self.q1
        elif action == 2:
            return self.q2
        elif action == 3:
            return self.q3
        elif action == 4:
            return self.q4
        elif action == 5:
            return self.q5
        elif action == 6:
            return self.q6

        return -999999.0

    def set(mut self, action: Int, value: Float64):
        if action == 0:
            self.q0 = value
        elif action == 1:
            self.q1 = value
        elif action == 2:
            self.q2 = value
        elif action == 3:
            self.q3 = value
        elif action == 4:
            self.q4 = value
        elif action == 5:
            self.q5 = value
        elif action == 6:
            self.q6 = value


# -----------------------------------------------------------------------------
# CNN evaluator
# -----------------------------------------------------------------------------

struct CnnEvaluator:
    var weights: CnnWeights

    def __init__(out self, var weights: CnnWeights):
        self.weights = weights^

    def predict_q_values(self, board: BinaryBoard) -> QValues7:
        var input = List[Float64]()
        for _ in range(INPUT_SIZE):
            input.append(0.0)

        encode_board(board, input)

        var h1 = List[Float64]()
        for _ in range(CONV1_OUT * BOARD_SIZE):
            h1.append(0.0)

        var h2 = List[Float64]()
        for _ in range(CONV2_OUT * BOARD_SIZE):
            h2.append(0.0)

        var hidden = List[Float64]()
        for _ in range(HIDDEN):
            hidden.append(0.0)

        conv1_same_relu(input, h1, self.weights)
        conv2_same_relu(h1, h2, self.weights)
        dense1_relu(h2, hidden, self.weights)

        var q = QValues7()

        for action in range(WIDTH):
            var sum = self.weights.fc2_b(action)

            for i in range(HIDDEN):
                sum += hidden[i] * self.weights.fc2_w(action, i)

            q.set(action, tanh_approx(sum))

        return q^


def encode_board(board: BinaryBoard, mut input: List[Float64]):
    for col in range(WIDTH):
        for row_from_bottom in range(HEIGHT):
            var bit_index = col * STRIDE + row_from_bottom
            var bit = UInt64(1) << UInt64(bit_index)

            # CNN row 0 = top row.
            var row = HEIGHT - 1 - row_from_bottom
            var base = row * WIDTH + col

            if (board.current_player_bits & bit) != UInt64(0):
                input[base] = 1.0

            if (board.opponent_bits & bit) != UInt64(0):
                input[BOARD_SIZE + base] = 1.0


def chw_index(ch: Int, row: Int, col: Int) -> Int:
    return ch * BOARD_SIZE + row * WIDTH + col


def relu(x: Float64) -> Float64:
    if x > 0.0:
        return x

    return 0.0


def tanh_approx(x: Float64) -> Float64:
    # Cheap monotonic squash into approximately [-1, 1].
    # This does not exactly match PyTorch tanh, but it is stable and fast.
    if x >= 0.0:
        return x / (1.0 + x)

    return x / (1.0 - x)


def conv1_same_relu(input: List[Float64], mut output: List[Float64], weights: CnnWeights):
    for out_ch in range(CONV1_OUT):
        for row in range(HEIGHT):
            for col in range(WIDTH):
                var sum = weights.conv1_b(out_ch)

                for in_ch in range(INPUT_CHANNELS):
                    for kr in range(3):
                        for kc in range(3):
                            var rr = row + kr - 1
                            var cc = col + kc - 1

                            if rr < 0 or rr >= HEIGHT:
                                continue

                            if cc < 0 or cc >= WIDTH:
                                continue

                            sum += input[chw_index(in_ch, rr, cc)] * weights.conv1_w(out_ch, in_ch, kr, kc)

                output[chw_index(out_ch, row, col)] = relu(sum)


def conv2_same_relu(input: List[Float64], mut output: List[Float64], weights: CnnWeights):
    for out_ch in range(CONV2_OUT):
        for row in range(HEIGHT):
            for col in range(WIDTH):
                var sum = weights.conv2_b(out_ch)

                for in_ch in range(CONV1_OUT):
                    for kr in range(3):
                        for kc in range(3):
                            var rr = row + kr - 1
                            var cc = col + kc - 1

                            if rr < 0 or rr >= HEIGHT:
                                continue

                            if cc < 0 or cc >= WIDTH:
                                continue

                            sum += input[chw_index(in_ch, rr, cc)] * weights.conv2_w(out_ch, in_ch, kr, kc)

                output[chw_index(out_ch, row, col)] = relu(sum)


def dense1_relu(input: List[Float64], mut output: List[Float64], weights: CnnWeights):
    for out_i in range(HIDDEN):
        var sum = weights.fc1_b(out_i)

        for in_i in range(FLAT_SIZE):
            sum += input[in_i] * weights.fc1_w(out_i, in_i)

        output[out_i] = relu(sum)


# -----------------------------------------------------------------------------
# Negamax search
# -----------------------------------------------------------------------------

struct NegamaxSearch:
    @staticmethod
    def choose_action(board: BinaryBoard, evaluator: CnnEvaluator, depth: Int) -> Int:
        if depth <= 0:
            return NegamaxSearch.choose_action_greedy(board, evaluator)

        var best_action = first_valid_action(board.valid_mask)
        var best_score = -999999.0

        for i in range(WIDTH):
            var action = move_order(i)

            if (board.valid_mask & (1 << action)) == 0:
                continue

            var child = board.play(action)

            var score = -NegamaxSearch.negamax(
                child,
                evaluator,
                depth - 1,
                -999999.0,
                999999.0,
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @staticmethod
    def choose_action_greedy(board: BinaryBoard, evaluator: CnnEvaluator) -> Int:
        var q = evaluator.predict_q_values(board)
        var best_action = first_valid_action(board.valid_mask)
        var best_q = q.get(best_action)

        for i in range(WIDTH):
            var action = move_order(i)

            if (board.valid_mask & (1 << action)) == 0:
                continue

            var value = q.get(action)

            if value > best_q:
                best_q = value
                best_action = action

        return best_action

    @staticmethod
    def negamax(
        board: BinaryBoard,
        evaluator: CnnEvaluator,
        depth: Int,
        alpha_in: Float64,
        beta: Float64,
    ) -> Float64:
        var terminal = board.terminal_value_for_current_player()

        if terminal != 2.0:
            return terminal

        if depth <= 0:
            return NegamaxSearch.evaluate_leaf(board, evaluator)

        var alpha = alpha_in
        var best = -999999.0

        for i in range(WIDTH):
            var action = move_order(i)

            if (board.valid_mask & (1 << action)) == 0:
                continue

            var child = board.play(action)

            var score = -NegamaxSearch.negamax(
                child,
                evaluator,
                depth - 1,
                -beta,
                -alpha,
            )

            if score > best:
                best = score

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        return best

    @staticmethod
    def evaluate_leaf(board: BinaryBoard, evaluator: CnnEvaluator) -> Float64:
        var q = evaluator.predict_q_values(board)
        var best_action = first_valid_action(board.valid_mask)
        var best_q = q.get(best_action)

        for i in range(WIDTH):
            var action = move_order(i)

            if (board.valid_mask & (1 << action)) == 0:
                continue

            var value = q.get(action)

            if value > best_q:
                best_q = value

        return best_q


def move_order(i: Int) -> Int:
    # Center-first order improves alpha-beta pruning for Connect 4.
    if i == 0:
        return 3
    elif i == 1:
        return 2
    elif i == 2:
        return 4
    elif i == 3:
        return 1
    elif i == 4:
        return 5
    elif i == 5:
        return 0

    return 6


def first_valid_action(valid_mask: Int) -> Int:
    for i in range(WIDTH):
        var action = move_order(i)

        if (valid_mask & (1 << action)) != 0:
            return action

    return 0
