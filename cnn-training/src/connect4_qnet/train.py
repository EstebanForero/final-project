from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from .dataset import QValuesDataset
from .export_raw import export_state_dict_raw
from .model import Connect4QNet


def state_split_indices(
    dataset: QValuesDataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    val_indices: list[int] = []
    threshold = int(val_fraction * 10_000)

    for index, record in enumerate(dataset.records):
        mixed = (record.state_key ^ (seed * 0x9E3779B97F4A7C15)) & ((1 << 128) - 1)
        mixed ^= mixed >> 64
        mixed = (mixed * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        bucket = mixed % 10_000

        if bucket < threshold:
            val_indices.append(index)
        else:
            train_indices.append(index)

    return train_indices, val_indices


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    visits: torch.Tensor,
) -> torch.Tensor:
    weights = torch.log1p(visits).view(-1, 1)
    loss = ((pred - target) ** 2) * mask * weights
    denom = (mask * weights).sum().clamp_min(1.0)
    return loss.sum() / denom


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def build_splits(
    dataset: QValuesDataset,
    *,
    val_fraction: float,
    split_mode: str,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    if val_fraction <= 0:
        return dataset, None

    if split_mode == "state":
        train_indices, val_indices = state_split_indices(dataset, val_fraction=val_fraction, seed=seed)
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        print(f"split=state train_samples={len(train_set)} val_samples={len(val_set)}")
    else:
        val_size = int(len(dataset) * val_fraction)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
        print(f"split=random train_samples={len(train_set)} val_samples={len(val_set)}")

    return train_set, val_set


def train_one_epoch(
    model: Connect4QNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"epoch {epoch}/{total_epochs}", unit="batch")
    for x, y, mask, visits in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        visits = visits.to(device, non_blocking=True)

        pred = model(x)
        loss = masked_weighted_mse(pred, y, mask, visits)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / max(len(loader), 1)


def evaluate(
    model: Connect4QNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y, mask, visits in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            visits = visits.to(device, non_blocking=True)

            pred = model(x)
            total_loss += masked_weighted_mse(pred, y, mask, visits).item()

    return total_loss / max(len(loader), 1)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)
    print(f"device={device} ({device_name})")

    dataset = QValuesDataset(args.q_values, min_visits=args.min_visits, max_samples=args.max_samples)
    print(f"samples={len(dataset)}")

    train_set, val_set = build_splits(
        dataset,
        val_fraction=args.val_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device) if val_set is not None else None

    model = Connect4QNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"epoch={epoch} train_loss={train_loss:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    save_file(state_dict, str(output))
    print(f"saved={output}")

    if args.raw_output:
        export_state_dict_raw(state_dict, Path(args.raw_output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-values", default="q_values", help="Path to Rust q_values file.")
    parser.add_argument("--output", default="connect4_qnet.safetensors")
    parser.add_argument("--raw-output", default="connect4_qnet.bin")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-visits", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--split-mode",
        choices=["state", "random"],
        default="state",
        help="Use state for unseen-board validation; random may leak states across splits.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
