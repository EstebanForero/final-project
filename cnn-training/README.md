# Connect 4 Q-network training

This folder trains a small CNN from the existing Rust `q_values` binary file.

Each record is 25 bytes:

```text
state_key: u128 little-endian
action: u8
q_value: f32 little-endian
visits: u32 little-endian
```

The trainer decodes `state_key` into a two-channel board tensor:

```text
channel 0 = current player's pieces
channel 1 = opponent's pieces
shape = [2, 6, 7]
```

## Quick start for Group C (Mojo)

The Mojo file (`groups/Group C/act_cnn_mojo.mojo`) reads raw weights from `connect4_qnet.bin`.
Training already produces this file automatically, so the full workflow is:

```bash
# 1. Train (outputs connect4_qnet.safetensors and connect4_qnet.bin)
uv run train-connect4-qnet --q-values q_values

# 2. Copy the raw weights to the Group C folder
cp connect4_qnet.bin "../groups/Group C/connect4_qnet.bin"
```

That's it — no separate export step needed.

---

## Inspect the data

```bash
uv run inspect-connect4-qvalues q_values --limit 200000
```

## Train

```bash
uv run train-connect4-qnet --q-values q_values --epochs 10 --batch-size 4096 --num-workers 4
```

The script uses CUDA automatically when `torch.cuda.is_available()` is true.
To force a device:

```bash
uv run train-connect4-qnet --device cuda
uv run train-connect4-qnet --device cpu
```

By default, validation uses a deterministic `state_key` split, so validation boards are not shared with training.

The output checkpoint defaults to:

```text
connect4_qnet.safetensors
```

The trainer also exports raw Rust-readable weights automatically by default:

```text
connect4_qnet.bin
```

## Export raw weights for Rust

This is only needed if you want to re-export an existing `.safetensors` file:

```bash
uv run export-connect4-qnet-raw --model-in connect4_qnet.safetensors --model-out connect4_qnet.bin
```

The raw binary is written as contiguous little-endian `f32` tensors in this order:

```text
conv1.weight, conv1.bias, conv2.weight, conv2.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias
```

Expected size:

```text
91479 f32 = 365916 bytes
```

## Evaluate generalization

Use this after training with the default `--split-mode state`:

```bash
uv run evaluate-connect4-qnet --model connect4_qnet.safetensors --q-values q_values --device cuda
```

The evaluator reports:

```text
record_metrics: MSE/RMSE/MAE on held-out state-action targets
state_policy_metrics: action agreement against the Q-table on unseen states
```

For a quick smoke run:

```bash
uv run train-connect4-qnet --q-values q_values --epochs 1 --max-samples 20000 --batch-size 1024
```
