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

EXPECTED_SHAPES = {
    "conv1.weight": (16, 2, 3, 3),
    "conv1.bias": (16,),
    "conv2.weight": (32, 16, 3, 3),
    "conv2.bias": (32,),
    "fc1.weight": (64, 32 * 6 * 7),
    "fc1.bias": (64,),
    "fc2.weight": (7, 64),
    "fc2.bias": (7,),
}

EXPECTED_FLOATS = 91_479
EXPECTED_BYTES = EXPECTED_FLOATS * 4


def export_raw(model_in: Path, model_out: Path) -> None:
    state = load_file(str(model_in))
    export_state_dict_raw(state, model_out)


def export_state_dict_raw(state: dict[str, torch.Tensor], model_out: Path) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)

    total_floats = 0
    with model_out.open("wb") as f:
        for name in ORDER:
            if name not in state:
                available = ", ".join(sorted(state))
                raise KeyError(f"Missing tensor {name!r}. Available tensors: {available}")

            tensor = state[name].detach().cpu().contiguous().to(torch.float32)
            shape = tuple(tensor.shape)
            expected_shape = EXPECTED_SHAPES[name]
            if shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {shape}, expected {expected_shape}. "
                    "Retrain with the current Connect4QNet architecture before exporting."
                )

            print(name, shape, tensor.numel())
            f.write(tensor.numpy().tobytes(order="C"))
            total_floats += tensor.numel()

    size = model_out.stat().st_size
    if total_floats != EXPECTED_FLOATS or size != EXPECTED_BYTES:
        raise ValueError(
            f"Invalid export size: floats={total_floats}, bytes={size}; "
            f"expected floats={EXPECTED_FLOATS}, bytes={EXPECTED_BYTES}"
        )

    print(f"saved {model_out}")
    print(f"floats={total_floats}")
    print(f"bytes={size}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", default="connect4_qnet.safetensors")
    parser.add_argument("--model-out", default="connect4_qnet.bin")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_raw(Path(args.model_in), Path(args.model_out))


if __name__ == "__main__":
    main()
