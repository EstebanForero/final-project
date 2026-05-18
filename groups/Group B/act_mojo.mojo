from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


comptime WIDTH = 7
comptime HEIGHT = 6
comptime STRIDE = HEIGHT + 1
comptime RECORD_SIZE = 25


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

    var action = GreedyPolicy.choose_action(
        q_store,
        encoded_state.state_key_current_as_a,
        encoded_state.state_key_current_as_b,
        board.valid_mask,
    )

    return action


fn get_q_values_cache() raises -> PythonObject:
    """
    Returns a process-wide Python dict used as a cache.

    The cache is stored on Python builtins to avoid needing mutable global Mojo
    state inside this extension module.
    """
    var builtins = Python.import_module("builtins")
    var cache_name = "_act_mojo_q_values_mmap_cache"
    var builtins_dict = builtins.__dict__

    if Bool(py=builtins_dict.__contains__(cache_name)):
        return builtins_dict.__getitem__(cache_name)

    var cache = builtins.dict()
    builtins_dict.__setitem__(cache_name, value=cache)
    return cache


fn get_cached_q_values_data(path_obj: PythonObject) raises -> PythonObject:
    """
    Returns a cached read-only mmap object for the Q-values file.

    This avoids:
      - reopening the same Q-values file every act(...) call
      - reading the entire file into a Python bytes object
      - duplicating the Q-table in memory
    """
    var builtins = Python.import_module("builtins")
    var mmap_module = Python.import_module("mmap")

    var path_key = builtins.str(path_obj)
    var cache = get_q_values_cache()

    if Bool(py=cache.__contains__(path_key)):
        return cache.__getitem__(path_key)

    var file = builtins.open(path_obj, "rb")

    var data = mmap_module.mmap(
        file.fileno(),
        0,
        access=mmap_module.ACCESS_READ,
    )

    file.close()

    cache.__setitem__(path_key, value=data)
    return data


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

        # Top row is indices 0..6.
        # If the top cell of a column is empty, that column is playable.
        for col in range(WIDTH):
            var top_value = Float64(py=flat_board.__getitem__(col))

            if top_value == 0.0:
                valid_mask |= 1 << col

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
        var data = get_cached_q_values_data(path_obj)
        var length = Int(py=data.__len__())

        # Python caches imported modules, so this is cheap.
        var struct_module = Python.import_module("struct")

        return QValueStore(data, length, struct_module)


struct ActionQValues:
    """
    Fixed-width action Q-value accumulator.

    This is similar in spirit to Rust's [ActionQValue; WIDTH], but avoids
    building a full HashMap<State, [ActionQValue; WIDTH]> in memory.

    Missing values remain 0.0, matching the behavior of the original lookup(...)
    method.
    """

    var seen_mask: Int

    var q0: Float64
    var q1: Float64
    var q2: Float64
    var q3: Float64
    var q4: Float64
    var q5: Float64
    var q6: Float64

    fn __init__(out self):
        self.seen_mask = 0

        self.q0 = 0.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.q4 = 0.0
        self.q5 = 0.0
        self.q6 = 0.0

    fn update(mut self, action: Int, q: Float64):
        """
        Updates the action value.

        If both state encodings match the same action, keep the larger Q-value.
        This preserves the old policy logic:

            q = max(q_a, q_b)
        """
        if (self.seen_mask & (1 << action)) == 0:
            self.set(action, q)
            self.seen_mask |= 1 << action
            return

        if q > self.get(action):
            self.set(action, q)

    fn get(self, action: Int) -> Float64:
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

        return 0.0

    fn set(mut self, action: Int, q: Float64):
        if action == 0:
            self.q0 = q
        elif action == 1:
            self.q1 = q
        elif action == 2:
            self.q2 = q
        elif action == 3:
            self.q3 = q
        elif action == 4:
            self.q4 = q
        elif action == 5:
            self.q5 = q
        elif action == 6:
            self.q6 = q


struct GreedyPolicy:
    @staticmethod
    fn choose_action(
        q_store: QValueStore,
        state_key_current_as_a: UInt128,
        state_key_current_as_b: UInt128,
        valid_mask: Int,
    ) raises -> Int:
        var values = ActionQValues()

        # Scan the Q-values table once.
        #
        # The original code called lookup(...) twice per valid action.
        # Each lookup(...) scanned the whole file, so a full board decision
        # could scan the Q-table up to 14 times.
        var offset = 0

        while offset + RECORD_SIZE <= q_store.length:
            var state_key = read_u128_le(q_store.data, offset)

            if state_key == state_key_current_as_a or state_key == state_key_current_as_b:
                var action = Int(py=q_store.data.__getitem__(offset + 16))

                if action >= 0 and action < WIDTH:
                    if (valid_mask & (1 << action)) != 0:
                        var q = read_f32_le(
                            q_store.struct_module,
                            q_store.data,
                            offset + 17,
                        )

                        values.update(action, q)

            offset += RECORD_SIZE

        var best_action = first_valid_action(valid_mask)
        var best_q = values.get(best_action)

        for action in range(WIDTH):
            if (valid_mask & (1 << action)) == 0:
                continue

            var q = values.get(action)

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
    var tuple_value = struct_module.unpack_from("<f", data, offset)
    return Float64(py=tuple_value.__getitem__(0))
