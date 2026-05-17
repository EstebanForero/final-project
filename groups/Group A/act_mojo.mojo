from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


@export
def PyInit_act_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_mojo")
        m.def_function[act]("act", docstring="Mojo policy action")
        return m.finalize()
    except e:
        abort(String("error creating Mojo module: ", e))


def act(board: PythonObject, rows_obj: PythonObject, cols_obj: PythonObject) raises -> PythonObject:
    # Temporary test: always center column.
    return PythonObject(3)
