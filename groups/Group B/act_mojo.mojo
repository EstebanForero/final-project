from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


comptime WIDTH = 7
comptime HEIGHT = 6
comptime STRIDE = HEIGHT + 1
comptime RECORD_SIZE = 25  # u128 + u8 + f32 + u32


@export
def PyInit_act_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_mojo")
        m.def_function[act]("act", docstring="Mojo greedy Q-value policy")
        return m.finalize()
    except e:
        abort(String("error creating Mojo module: ", e))


def act(
    flat_board: PythonObject,
    rows_obj: PythonObject,
    cols_obj: PythonObject,
    timeout_obj: PythonObject,
    q_values_path_obj: PythonObject,
) raises -> PythonObject:
    var board = BinaryBoard.from_python_board(flat_board)

    if board.valid_mask == 0:
        return 0

    var encoded_state = EncodedState.from_binary_board(board)
    var q_store = QValueStore.from_file(q_values_path_obj)

    var policy = GreedyPolicy()
    var action = policy.choose_action(
        q_store,
        encoded_state.state_key_current_as_a,
        encoded_state.state_key_current_as_b,
        board.valid_mask,
    )

    return action


struct BinaryBoard:
    var current_player_bits: UInt64
    var opponent_bits: UInt64
    var valid_mask: Int

    fn __init__(
        out self,
        current_player_bits: UInt64,
        opponent_bits: UInt64,
        valid_mask: Int,
    ):
        self.current_player_bits = current_player_bits
        self.opponent_bits = opponent_bits
        self.valid_mask = valid_mask

    @staticmethod
    fn from_python_board(flat_board: PythonObject) raises -> BinaryBoard:
        var current_player_bits = UInt64(0)
        var opponent_bits = UInt64(0)
        var valid_mask = 0

        # A column is valid if its top cell is empty.
        # Flat board is row-major, so top cell of column col is index col.
        for col in range(WIDTH):
            var top_value = Float64(py=flat_board.__getitem__(col))

            if top_value == 0.0:
                valid_mask |= 1 << col

        # Rust-compatible bitboard:
        #
        # bit_index = col * STRIDE + bit_row
        #
        # Python/NumPy board:
        #   row 0 = top
        #
        # Rust bitboard:
        #   bit_row 0 = bottom
        #
        # Therefore:
        #   bit_row = HEIGHT - 1 - row
        #
        # Assumption:
        #   value > 0  => current player's piece
        #   value < 0  => opponent's piece
        #   value == 0 => empty
        for row in range(HEIGHT):
            for col in range(WIDTH):
                var flat_index = row * WIDTH + col
                var value = Float64(py=flat_board.__getitem__(flat_index))

                if value == 0.0:
                    continue

                var bit_row = HEIGHT - 1 - row
                var bit_index = col * STRIDE + bit_row
                var bit = UInt64(1) << UInt64(bit_index)

                if value > 0.0:
                    current_player_bits |= bit
                else:
                    opponent_bits |= bit

        return BinaryBoard(current_player_bits, opponent_bits, valid_mask)


struct EncodedState:
    var state_key_current_as_a: UInt128
    var state_key_current_as_b: UInt128

    fn __init__(
        out self,
        state_key_current_as_a: UInt128,
        state_key_current_as_b: UInt128,
    ):
        self.state_key_current_as_a = state_key_current_as_a
        self.state_key_current_as_b = state_key_current_as_b

    @staticmethod
    fn from_binary_board(board: BinaryBoard) -> EncodedState:
        # Rust encoding:
        #
        #   a | (b << 64) | (player << 127)
        #
        # Because the tournament board is assumed to be current-player-relative,
        # we check two equivalent interpretations:
        #
        # 1. current player as Rust Player::A:
        #      A bits = current_player_bits
        #      B bits = opponent_bits
        #      current_player = 0
        #
        # 2. current player as Rust Player::B:
        #      A bits = opponent_bits
        #      B bits = current_player_bits
        #      current_player = 1
        var key_as_a = encode_state(
            board.current_player_bits,
            board.opponent_bits,
            UInt8(0),
        )

        var key_as_b = encode_state(
            board.opponent_bits,
            board.current_player_bits,
            UInt8(1),
        )

        return EncodedState(key_as_a, key_as_b)


struct QValueStore:
    var data: PythonObject
    var length: Int
    var struct_module: PythonObject

    fn __init__(
        out self,
        data: PythonObject,
        length: Int,
        struct_module: PythonObject,
    ):
        self.data = data
        self.length = length
        self.struct_module = struct_module

    @staticmethod
    fn from_file(path_obj: PythonObject) raises -> QValueStore:
        var builtins = Python.import_module("builtins")
        var file = builtins.open(path_obj, "rb")
        var data = file.read()
        file.close()

        var length = Int(py=data.__len__())
        var struct_module = Python.import_module("struct")

        return QValueStore(data, length, struct_module)

    fn lookup(self, target_state_key: UInt128, target_action: UInt8) raises -> Float64:
        var offset = 0

        while offset + RECORD_SIZE <= self.length:
            var state_key = read_u128_le(self.data, offset)
            var action = UInt8(Int(py=self.data.__getitem__(offset + 16)))

            if state_key == target_state_key and action == target_action:
                return read_f32_le(self.struct_module, self.data, offset + 17)

            offset += RECORD_SIZE

        return 0.0


struct GreedyPolicy:
    fn __init__(out self):
        pass

    fn choose_action(
        self,
        q_store: QValueStore,
        state_key_current_as_a: UInt128,
        state_key_current_as_b: UInt128,
        valid_mask: Int,
    ) raises -> Int:
        var best_action = first_valid_action(valid_mask)
        var best_q = -1000000000000000000000000000000.0

        for action in range(WIDTH):
            if (valid_mask & (1 << action)) == 0:
                continue

            var action_u8 = UInt8(action)

            var q_a = q_store.lookup(state_key_current_as_a, action_u8)
            var q_b = q_store.lookup(state_key_current_as_b, action_u8)

            var q = q_a
            if q_b > q:
                q = q_b

            if q > best_q:
                best_q = q
                best_action = action

        return best_action


fn encode_state(
    player_a_bits: UInt64,
    player_b_bits: UInt64,
    current_player: UInt8,
) -> UInt128:
    var a = UInt128(player_a_bits)
    var b = UInt128(player_b_bits)
    var p = UInt128(current_player)

    return a | (b << 64) | (p << 127)


fn first_valid_action(valid_mask: Int) -> Int:
    for action in range(WIDTH):
        if (valid_mask & (1 << action)) != 0:
            return action

    return 0


fn read_u128_le(data: PythonObject, offset: Int) raises -> UInt128:
    var value = UInt128(0)

    for i in range(16):
        var byte = UInt128(Int(py=data.__getitem__(offset + i)))
        value |= byte << UInt128(8 * i)

    return value


fn read_f32_le(
    struct_module: PythonObject,
    data: PythonObject,
    offset: Int,
) raises -> Float64:
    # Equivalent to Python:
    #
    #   struct.unpack_from("<f", data, offset)[0]
    #
    # This avoids depending on Mojo bitcast syntax.
    var tuple_value = struct_module.unpack_from("<f", data, offset)
    return Float64(py=tuple_value.__getitem__(0))
