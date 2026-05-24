from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort

comptime WIDTH  = 7
comptime HEIGHT = 6


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
