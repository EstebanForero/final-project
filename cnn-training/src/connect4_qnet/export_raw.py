from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

ORDER = [
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
]

# ── Q-value model (Connect4QNet): fc2 outputs 7 values ───────────────────────
Q_SHAPES = {
    "conv1.weight": (16, 2, 3, 3),
    "conv1.bias":   (16,),
    "conv2.weight": (32, 16, 3, 3),
    "conv2.bias":   (32,),
    "fc1.weight":   (64, 32 * 6 * 7),
    "fc1.bias":     (64,),
    "fc2.weight":   (7, 64),
    "fc2.bias":     (7,),
}
Q_FLOATS = 91_479
Q_BYTES  = Q_FLOATS * 4

# ── V-value model (Connect4VNet): fc2 outputs 1 value ────────────────────────
V_SHAPES = {
    "conv1.weight": (16, 2, 3, 3),
    "conv1.bias":   (16,),
    "conv2.weight": (32, 16, 3, 3),
    "conv2.bias":   (32,),
    "fc1.weight":   (64, 32 * 6 * 7),
    "fc1.bias":     (64,),
    "fc2.weight":   (1, 64),
    "fc2.bias":     (1,),
}
V_FLOATS = 91_089   # Q_FLOATS - 7*64 - 7 + 1*64 + 1 = 91_479 - 390 = 91_089
V_BYTES  = V_FLOATS * 4

# Keep old names for backwards compatibility
EXPECTED_SHAPES = Q_SHAPES
EXPECTED_FLOATS = Q_FLOATS
EXPECTED_BYTES  = Q_BYTES


def _detect_shapes(state: dict[str, torch.Tensor]) -> tuple[dict, int, int]:
    """Auto-detect whether this is a Q-model or V-model from fc2 shape."""
    fc2_shape = tuple(state["fc2.weight"].shape)
    if fc2_shape == (1, 64):
        return V_SHAPES, V_FLOATS, V_BYTES
    if fc2_shape == (7, 64):
        return Q_SHAPES, Q_FLOATS, Q_BYTES
    raise ValueError(f"Unexpected fc2.weight shape {fc2_shape}; expected (1,64) or (7,64).")


def export_raw(model_in: Path, model_out: Path) -> None:
    state = load_file(str(model_in))
    export_state_dict_raw(state, model_out)


def export_state_dict_raw(state: dict[str, torch.Tensor], model_out: Path) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    expected_shapes, expected_floats, expected_bytes = _detect_shapes(state)

    total_floats = 0
    with model_out.open("wb") as f:
        for name in ORDER:
            if name not in state:
                available = ", ".join(sorted(state))
                raise KeyError(f"Missing tensor {name!r}. Available tensors: {available}")

            tensor = state[name].detach().cpu().contiguous().to(torch.float32)
            shape  = tuple(tensor.shape)
            expected_shape = expected_shapes[name]
            if shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {shape}, expected {expected_shape}."
                )

            print(name, shape, tensor.numel())
            f.write(tensor.numpy().tobytes(order="C"))
            total_floats += tensor.numel()

    size = model_out.stat().st_size
    if total_floats != expected_floats or size != expected_bytes:
        raise ValueError(
            f"Invalid export size: floats={total_floats}, bytes={size}; "
            f"expected floats={expected_floats}, bytes={expected_bytes}"
        )

    print(f"saved {model_out}")
    print(f"floats={total_floats}")
    print(f"bytes={size}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in",  default="connect4_qnet.safetensors")
    parser.add_argument("--model-out", default="connect4_qnet.bin")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_raw(Path(args.model_in), Path(args.model_out))


if __name__ == "__main__":
    main()
