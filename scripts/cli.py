import os
import csv
import math
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity

from PhotonNet import TemporalAttentionUNet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_pairs(data_root):
    data_root = Path(data_root)
    pairs = []
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        npy_files = sorted(class_dir.glob("*.npy"))
        for npy_path in npy_files:
            png_path = npy_path.with_suffix(".png")
            if png_path.exists():
                pairs.append((str(npy_path), str(png_path)))
    return pairs


class SPCDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _load_target(self, png_path):
        img = Image.open(png_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return arr

    def _make_8_groups(self, npy_path):
        photoncube = np.load(npy_path, mmap_mode="r")  # (1024, 800, 100, 3)
        parts = np.zeros((8, 800, 800, 3), dtype=np.uint16)

        frames_per_group = photoncube.shape[0] // 8  # 128
        for i in range(8):
            chunk = photoncube[i * frames_per_group:(i + 1) * frames_per_group]   # (128,800,100,3)
            chunk_bits = np.unpackbits(chunk, axis=2)                             # (128,800,800,3)
            parts[i] = chunk_bits.sum(axis=0, dtype=np.uint16)                    # (800,800,3)

        parts = parts.astype(np.float32)
        parts /= (parts.max() + 1e-8)
        parts = torch.from_numpy(parts)  # (8,800,800,3)
        return parts

    def __getitem__(self, idx):
        npy_path, png_path = self.pairs[idx]
        x = self._make_8_groups(npy_path)
        y = self._load_target(png_path)
        return x, y, npy_path, png_path


def collate_fn(batch):
    xs, ys, npy_paths, png_paths = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,8,800,800,3)
    y = torch.stack(ys, dim=0)  # (B,3,800,800)
    return x, y, list(npy_paths), list(png_paths)


def create_dataloaders(data_path, batch_size, num_workers, seed):
    pairs = find_pairs(data_path)
    if len(pairs) == 0:
        raise RuntimeError(f"No (.npy, .png) pairs found under {data_path}")

    dataset = SPCDataset(pairs)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    return train_loader, val_loader, len(dataset), len(train_ds), len(val_ds)


def compute_metrics(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    psnr_each = peak_signal_noise_ratio(pred, target, data_range=1.0, reduction="none")
    ms_ssim_each = multiscale_structural_similarity_index_measure(
        pred, target, data_range=1.0, reduction="none"
    )
    lpips_each = learned_perceptual_image_patch_similarity(
        pred, target, net_type="alex", normalize=True
    )

    psnr_each = psnr_each.reshape(-1)
    ms_ssim_each = ms_ssim_each.reshape(-1)
    lpips_each = lpips_each.reshape(-1)

    return psnr_each.detach(), ms_ssim_each.detach(), lpips_each.detach()


def summarize_metric(values, higher_is_better=True):
    vals = torch.cat([v.reshape(-1) for v in values]).float().cpu()
    mean_val = vals.mean().item()

    n = max(1, math.ceil(len(vals) * 0.05))
    m = max(1, math.ceil(len(vals) * 0.01))

    sorted_vals, _ = torch.sort(vals)
    if higher_is_better:
        low_5 = sorted_vals[:n].mean().item()
        low_1 = sorted_vals[:m].mean().item()
    else:
        low_5 = sorted_vals[-n:].mean().item()
        low_1 = sorted_vals[-m:].mean().item()

    return mean_val, low_5, low_1


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_csv_row(csv_path, row, header=None):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint_for_pretraining(model, ckpt_path, device, strict=False):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    load_result = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        print(f"Loaded pretrained weights from: {ckpt_path}")
        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)
    else:
        print(f"Loaded pretrained weights strictly from: {ckpt_path}")

    return model


def resume_from_checkpoint(model, optimizer, scheduler, scaler, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
    best_lpips = checkpoint.get("metrics", {}).get("mean_lpips", float("inf"))

    print(f"Resumed training from checkpoint: {ckpt_path}")
    print(f"Starting at epoch: {start_epoch}")

    return model, optimizer, scheduler, scaler, start_epoch, best_val_loss, best_lpips


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, epochs, grad_accum_steps, heatmap_loss_weight):
    model.train()
    running_loss = 0.0
    num_steps = 0

    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y, _, _) in enumerate(pbar, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            recon = out["recon"]
            heatmap = out["heatmap"]

            loss_recon = F.l1_loss(recon, y)
            loss_heat = F.l1_loss(heatmap, y)
            loss = loss_recon + heatmap_loss_weight * loss_heat
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * grad_accum_steps
        num_steps += 1
        pbar.set_postfix(loss=f"{running_loss / num_steps:.5f}")

    return running_loss / max(1, num_steps)


@torch.no_grad()
def validate(model, loader, device, epoch, epochs, heatmap_loss_weight):
    model.eval()
    running_loss = 0.0
    num_steps = 0

    all_psnr = []
    all_ms_ssim = []
    all_lpips = []

    pbar = tqdm(loader, desc=f"Val {epoch}/{epochs}", leave=False)

    for x, y, _, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            recon = out["recon"]
            heatmap = out["heatmap"]

            loss_recon = F.l1_loss(recon, y)
            loss_heat = F.l1_loss(heatmap, y)
            loss = loss_recon + heatmap_loss_weight * loss_heat

        running_loss += loss.item()
        num_steps += 1

        pred_metrics = recon.float()
        target_metrics = y.float()

        psnr_each, ms_ssim_each, lpips_each = compute_metrics(pred_metrics, target_metrics)
        all_psnr.append(psnr_each)
        all_ms_ssim.append(ms_ssim_each)
        all_lpips.append(lpips_each)

        pbar.set_postfix(loss=f"{running_loss / num_steps:.5f}")

    mean_psnr, low5_psnr, low1_psnr = summarize_metric(all_psnr, higher_is_better=True)
    mean_msssim, low5_msssim, low1_msssim = summarize_metric(all_ms_ssim, higher_is_better=True)
    mean_lpips, low5_lpips, low1_lpips = summarize_metric(all_lpips, higher_is_better=False)

    metrics = {
        "val_loss": running_loss / max(1, num_steps),
        "mean_psnr": mean_psnr,
        "low5_psnr": low5_psnr,
        "low1_psnr": low1_psnr,
        "mean_ms_ssim": mean_msssim,
        "low5_ms_ssim": low5_msssim,
        "low1_ms_ssim": low1_msssim,
        "mean_lpips": mean_lpips,
        "low5_lpips": low5_lpips,
        "low1_lpips": low1_lpips,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to spc/train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./runs_spc")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--unet_base", type=int, default=32)
    parser.add_argument("--heatmap_loss_weight", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Path to checkpoint for initializing model weights")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to full checkpoint to resume training")
    parser.add_argument("--strict_load", action="store_true",
                        help="Use strict=True when loading pretrained weights")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"
    sample_dir = save_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), log_dir / "config.json")

    train_loader, val_loader, n_total, n_train, n_val = create_dataloaders(
        args.data_path, args.batch_size, args.num_workers, args.seed
    )

    model = TemporalAttentionUNet(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        unet_base=args.unet_base
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_val_loss = float("inf")
    best_lpips = float("inf")
    metrics_csv = log_dir / "metrics.csv"

    meta = {
        "num_total": n_total,
        "num_train": n_train,
        "num_val": n_val,
        "device": str(device)
    }
    save_json(meta, log_dir / "dataset_info.json")

    if args.resume_ckpt is not None and args.pretrained_ckpt is not None:
        raise ValueError("Use only one of --resume_ckpt or --pretrained_ckpt")

    if args.resume_ckpt is not None:
        model, optimizer, scheduler, scaler, start_epoch, best_val_loss, best_lpips = resume_from_checkpoint(
            model, optimizer, scheduler, scaler, args.resume_ckpt, device
        )
    elif args.pretrained_ckpt is not None:
        model = load_checkpoint_for_pretraining(
            model, args.pretrained_ckpt, device, strict=args.strict_load
        )

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, args.epochs, args.grad_accum_steps, args.heatmap_loss_weight
        )

        val_metrics = validate(
            model, val_loader, device, epoch, args.epochs, args.heatmap_loss_weight
        )

        scheduler.step()
        epoch_time = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "mean_psnr": val_metrics["mean_psnr"],
            "low5_psnr": val_metrics["low5_psnr"],
            "low1_psnr": val_metrics["low1_psnr"],
            "mean_ms_ssim": val_metrics["mean_ms_ssim"],
            "low5_ms_ssim": val_metrics["low5_ms_ssim"],
            "low1_ms_ssim": val_metrics["low1_ms_ssim"],
            "mean_lpips": val_metrics["mean_lpips"],
            "low5_lpips": val_metrics["low5_lpips"],
            "low1_lpips": val_metrics["low1_lpips"],
            "epoch_time_sec": epoch_time
        }
        append_csv_row(metrics_csv, row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_metrics['val_loss']:.5f} | "
            f"Mean PSNR={val_metrics['mean_psnr']:.4f} | "
            f"5% Low PSNR={val_metrics['low5_psnr']:.4f} | "
            f"1% Low PSNR={val_metrics['low1_psnr']:.4f} | "
            f"Mean MS-SSIM={val_metrics['mean_ms_ssim']:.4f} | "
            f"5% Low MS-SSIM={val_metrics['low5_ms_ssim']:.4f} | "
            f"1% Low MS-SSIM={val_metrics['low1_ms_ssim']:.4f} | "
            f"Mean LPIPS={val_metrics['mean_lpips']:.4f} | "
            f"5% Low LPIPS={val_metrics['low5_lpips']:.4f} | "
            f"1% Low LPIPS={val_metrics['low1_lpips']:.4f}"
        )

        latest_ckpt = ckpt_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "metrics": row
        }, latest_ckpt)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "metrics": row
            }, ckpt_dir / "best_val_loss.pt")

        if val_metrics["mean_lpips"] < best_lpips:
            best_lpips = val_metrics["mean_lpips"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "metrics": row
            }, ckpt_dir / "best_lpips.pt")


if __name__ == "__main__":
    main()
