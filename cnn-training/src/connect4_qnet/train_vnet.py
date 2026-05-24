from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from .dataset import VValuesDataset
from .export_raw import export_state_dict_raw
from .model import Connect4VNet


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, visits: torch.Tensor) -> torch.Tensor:
    weights = torch.log1p(visits)
    loss = ((pred - target.squeeze(1)) ** 2) * weights.squeeze(1)
    return loss.sum() / weights.sum().clamp_min(1.0)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
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


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"epoch {epoch}/{total_epochs}", unit="batch")
    for x, y, w in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        pred = model(x)
        loss = weighted_mse(pred, y, w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.5f}")
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y, w in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            pred = model(x)
            total_loss += weighted_mse(pred, y, w).item()
    return total_loss / max(len(loader), 1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)
    print(f"device={device} ({device_name})")

    dataset = VValuesDataset(args.q_values, min_visits=args.min_visits, max_samples=args.max_samples)
    print(f"unique_states={len(dataset)}")

    if args.val_fraction > 0:
        val_size = int(len(dataset) * args.val_fraction)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"train={train_size} val={val_size}")
    else:
        train_set, val_set = dataset, None

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device) if val_set else None

    model = Connect4VNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        if val_loader:
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
    parser.add_argument("--q-values", default="q_values")
    parser.add_argument("--output", default="connect4_vnet.safetensors")
    parser.add_argument("--raw-output", default="connect4_vnet.bin")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-visits", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
