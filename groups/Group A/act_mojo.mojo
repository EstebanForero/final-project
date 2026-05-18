from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.time import perf_counter_ns


@export
def PyInit_act_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("act_mojo")
        m.def_function[act]("act", docstring="Mojo policy action")
        return m.finalize()
    except e:
        abort(String("error creating Mojo module: ", e))


def act(
    board: PythonObject,
    rows_obj: PythonObject,
    cols_obj: PythonObject,
    timeout_obj: PythonObject,
) raises -> PythonObject:
    # Realistic small CNN benchmark.
    #
    # Keep this small first.
    # If it runs safely, increase calls/forwards_per_call.
    var calls = 5
    var forwards_per_call = 2

    # Warmup.
    var warmup_acc = 0.0
    for _ in range(2):
        warmup_acc += run_many_forwards(forwards_per_call)

    var start = perf_counter_ns()

    var acc = 0.0
    for _ in range(calls):
        acc += run_many_forwards(forwards_per_call)

    var end = perf_counter_ns()

    var total_ns = Float64(end - start)
    var total_forwards = calls * forwards_per_call

    var total_ms = total_ns / 1_000_000.0
    var avg_call_us = total_ns / Float64(calls) / 1_000.0
    var avg_forward_us = total_ns / Float64(total_forwards) / 1_000.0

    # Approximate multiply-add workload:
    #
    # Conv1:
    #   6*7*16*2*3*3 = 12096 MACs
    #
    # Conv2:
    #   6*7*32*16*3*3 = 193536 MACs
    #
    # Dense1:
    #   1344*64 = 86016 MACs
    #
    # Dense2:
    #   64*1 = 64 MACs
    #
    # Total:
    #   291712 MACs per forward
    var approx_macs_per_forward = 291712
    var approx_total_macs = approx_macs_per_forward * total_forwards

    var report = String()
    report += "MOJO_STEP_4_REALISTIC_SMALL_CNN_BENCH_OK\n"
    report += "architecture=2x6x7 -> conv2to16 -> conv16to32 -> dense1344to64 -> dense64to1\n"
    report += "calls=" + String(calls) + "\n"
    report += "forwards_per_call=" + String(forwards_per_call) + "\n"
    report += "total_forwards=" + String(total_forwards) + "\n"
    report += "approx_macs_per_forward=" + String(approx_macs_per_forward) + "\n"
    report += "approx_total_macs=" + String(approx_total_macs) + "\n"
    report += "total_ms=" + String(total_ms) + "\n"
    report += "avg_call_us=" + String(avg_call_us) + "\n"
    report += "avg_forward_us=" + String(avg_forward_us) + "\n"
    report += "acc=" + String(acc) + "\n"
    report += "warmup_acc=" + String(warmup_acc)

    return PythonObject(report)


def run_many_forwards(reps: Int) -> Float64:
    var acc = 0.0

    for i in range(reps):
        acc += realistic_small_cnn_forward(Float64(i) * 0.001)

    return acc


def relu(x: Float64) -> Float64:
    if x > 0.0:
        return x
    return 0.0


def realistic_small_cnn_forward(seed: Float64) -> Float64:
    # Simulated arithmetic workload for this value network:
    #
    # Input: 2 × 6 × 7
    #
    # Conv1:
    #   in_channels=2
    #   out_channels=16
    #   kernel=3x3
    #
    # Conv2:
    #   in_channels=16
    #   out_channels=32
    #   kernel=3x3
    #
    # Dense:
    #   1344 -> 64
    #   64 -> 1
    #
    # This does not allocate real tensors yet.
    # It approximates the CPU arithmetic cost.

    var acc = seed

    # -----------------------
    # Conv1-like workload
    # 2 -> 16 channels
    # -----------------------
    for out_ch in range(16):
        for r in range(6):
            for c in range(7):
                var s = 0.01 * Float64(out_ch + 1)

                for in_ch in range(2):
                    for kr in range(3):
                        for kc in range(3):
                            var feature = (
                                seed
                                + Float64(in_ch + 1) * 0.01
                                + Float64((r + 1) * (c + 1)) * 0.001
                            )

                            var w = (
                                0.001
                                * Float64(out_ch + 1)
                                * Float64(in_ch + 1)
                                * Float64(kr + 1)
                                * Float64(kc + 1)
                            )

                            s += feature * w

                acc += relu(s)

    # -----------------------
    # Conv2-like workload
    # 16 -> 32 channels
    # -----------------------
    for out_ch in range(32):
        for r in range(6):
            for c in range(7):
                var s = 0.01 * Float64(out_ch + 1)

                for in_ch in range(16):
                    for kr in range(3):
                        for kc in range(3):
                            var feature = (
                                acc * 0.000001
                                + Float64(in_ch + 1) * 0.0001
                                + Float64((r + 1) * (c + 1)) * 0.00001
                            )

                            var w = (
                                0.0001
                                * Float64(out_ch + 1)
                                * Float64(in_ch + 1)
                                * Float64(kr + 1)
                                * Float64(kc + 1)
                            )

                            s += feature * w

                acc += relu(s)

    # -----------------------
    # Dense 1344 -> 64
    # -----------------------
    for hidden in range(64):
        var s = 0.001 * Float64(hidden + 1)

        for i in range(1344):
            var feature = acc * 0.000001 + Float64((i % 17) + 1) * 0.0001
            var w = 0.00001 * Float64((hidden % 11) + 1) * Float64((i % 13) + 1)

            s += feature * w

        acc += relu(s)

    # -----------------------
    # Dense 64 -> 1
    # -----------------------
    var out = 0.0
    for i in range(64):
        var feature = acc * 0.000001 + Float64(i + 1) * 0.0001
        var w = 0.001 * Float64((i % 7) + 1)
        out += feature * w

    # Tanh-ish squash into approximately [-1, 1].
    if out >= 0.0:
        return out / (1.0 + out)
    else:
        return out / (1.0 - out)
