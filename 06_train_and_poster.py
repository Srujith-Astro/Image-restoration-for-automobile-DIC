"""
06_train_and_poster.py -- run this ON YOUR PC, from inside the SALAAR_KLA_PS01
repo root (it imports models/ and utils/ from there).

Works directly with your actual folders:
    train_GT       = clean ground truth
    train_NoisyLR  = degraded/noisy low-res counterpart (same filenames)
    test_NoisyLR   = real hidden test set, NO ground truth (qualitative only)

It:
  1. carves a held-out val split out of train_GT/train_NoisyLR (no separate
     val folder needed),
  2. trains U-Net and NAFNet (same data pipeline, same loss, same optimizer/
     schedule -- apples-to-apples, same idea as the repo's train.py),
  3. saves a PSNR/SSIM-vs-epoch curve for both models on one chart,
  4. saves a GT | Noisy (upsampled) | U-Net restored | NAFNet restored panel
     with per-image PSNR, for a handful of val images,
  5. if --test_lr is given, also renders restored-only panels on a few real
     test_NoisyLR images (no GT available for those, so no PSNR there).

Usage (defaults are tuned to finish fast on a good GPU; raise --epochs for
better numbers once you've seen this run end-to-end):

    python 06_train_and_poster.py ^
        --train_gt "F:\Hack\kla hack 3\train_GT" ^
        --train_lr "F:\Hack\kla hack 3\train_NoisyLR" ^
        --test_lr  "F:\Hack\kla hack 3\test_NoisyLR" ^
        --epochs 20 --out_dir poster_run

Everything lands in --out_dir:
    poster_run/curves.png              <- PSNR/SSIM vs epoch, both models
    poster_run/comparison_panel.png    <- GT | Noisy | U-Net | NAFNet
    poster_run/test_panel.png          <- real test_NoisyLR, restored-only
    poster_run/unet_best.pt, poster_run/nafnet_best.pt
    poster_run/history.json            <- raw numbers, in case you want a table
"""
import argparse, os, sys, json, random, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(REPO_ROOT, "models")):
    REPO_ROOT = os.path.join(REPO_ROOT, "SALAAR_KLA_PS01")
sys.path.insert(0, REPO_ROOT)
from models.unet import build_unet
from models.nafnet import build_nafnet
from utils.dataset import PairedRestorationDataset, _list_images, _load_array
from utils.losses import build_loss
from utils.metrics import psnr, ssim, count_params

SEED = 42


def build_model(name, size, device):
    fn = build_unet if name == "unet" else build_nafnet
    return fn(in_channels=1, out_channels=1, size=size).to(device)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training, but this PyTorch environment cannot access a GPU. "
            "Install a CUDA-enabled PyTorch build and try again."
        )
    torch.cuda.set_device(0)
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA preflight passed: cuda:0 ({device_name})")
    return "cuda"


def split_indices(n, val_frac, seed=SEED):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]  # train_idx, val_idx


@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    ps, ss = [], []
    for batch in loader:
        lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        pred = model(lr).clamp(0, 1)
        ps.append(psnr(pred, hr)); ss.append(ssim(pred, hr))
    return sum(ps) / len(ps), sum(ss) / len(ss)


def train_model(name, args, train_set, val_set, device):
    model = build_model(name, args.size, device)
    print(f"\n[{name}] params: {count_params(model)/1e6:.2f}M | device: {device}")

    loss_stage = args.unet_loss if name == "unet" else args.nafnet_loss
    loss_fn = build_loss({"loss_stage": loss_stage}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    history, best_psnr = [], -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for batch in train_loader:
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            opt.zero_grad()
            pred = model(lr)
            loss, _ = loss_fn(pred, hr)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        val_psnr, val_ssim = run_validation(model, val_loader, device)
        row = dict(epoch=ep, train_loss=running / len(train_loader),
                   val_psnr=val_psnr, val_ssim=val_ssim, time_s=round(time.time() - t0, 1))
        history.append(row)
        print(f"[{name}] epoch {ep:03d}/{args.epochs} | loss {row['train_loss']:.4f} | "
              f"val_psnr {val_psnr:.2f} | val_ssim {val_ssim:.4f} | {row['time_s']:.1f}s")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{name}_best.pt"))
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f"{name}_best.pt"), map_location=device))
    return model, history


def plot_curves(histories, out_path):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        ep = [r["epoch"] for r in hist]
        plt.plot(ep, [r["val_psnr"] for r in hist], label=f"{name} val PSNR")
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR during training"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved {out_path}")


@torch.no_grad()
def render_comparison_panel(models, val_set, indices, out_path, device):
    ncols = 2 + len(models)
    fig, axes = plt.subplots(len(indices), ncols, figsize=(3.2 * ncols, 3.2 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]
    for row, i in enumerate(indices):
        sample = val_set[i]
        lr, hr = sample["lr"].unsqueeze(0).to(device), sample["hr"].unsqueeze(0).to(device)
        lr_np, hr_np = lr.squeeze().cpu().numpy(), hr.squeeze().cpu().numpy()
        axes[row, 0].imshow(hr_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Ground truth" if row == 0 else "")
        axes[row, 1].imshow(lr_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Noisy (upsampled)" if row == 0 else "")
        col = 2
        for name, model in models.items():
            pred = model(lr).clamp(0, 1).squeeze().cpu().numpy()
            p = psnr(model(lr).clamp(0, 1), hr)
            axes[row, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{name} restored" if row == 0 else "")
            axes[row, col].set_xlabel(f"PSNR {p:.2f} dB", fontsize=8)
            col += 1
        for c in range(ncols):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    print(f"Saved {out_path}")


@torch.no_grad()
def render_test_panel(models, test_lr_dir, n, out_path, device, upsample_lr=True):
    paths = _list_images(test_lr_dir)[:n]
    if not paths:
        print(f"[warn] no images found in {test_lr_dir}, skipping test panel"); return
    ncols = 1 + len(models)
    fig, axes = plt.subplots(len(paths), ncols, figsize=(3.2 * ncols, 3.2 * len(paths)))
    if len(paths) == 1:
        axes = axes[None, :]
    for row, p in enumerate(paths):
        arr = _load_array(p)
        if upsample_lr:
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        axes[row, 0].imshow(arr, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Real test input (noisy)" if row == 0 else "")
        col = 1
        for name, model in models.items():
            pred = model(x).clamp(0, 1).squeeze().cpu().numpy()
            axes[row, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{name} restored" if row == 0 else "")
            col += 1
        for c in range(ncols):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_gt", required=True)
    ap.add_argument("--train_lr", required=True)
    ap.add_argument("--test_lr", default=None)
    ap.add_argument("--out_dir", default="poster_run")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", choices=["tiny", "small", "base"], default="small")
    ap.add_argument("--unet_loss", choices=["l1", "l1_ssim", "l1_ssim_edge"], default="l1")
    ap.add_argument("--nafnet_loss", choices=["l1", "l1_ssim", "l1_ssim_edge"], default="l1_ssim")
    ap.add_argument("--upsample_lr", action="store_true", default=True,
                     help="bicubic-upsample LR to HR size before crop (leave on -- this task is 2x)")
    ap.add_argument("--n_panel", type=int, default=5, help="how many val images in the comparison panel")
    ap.add_argument("--n_test_panel", type=int, default=4, help="how many real test images to render")
    args = ap.parse_args()

    device = require_cuda()
    torch.manual_seed(SEED); random.seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device}")

    train_full = PairedRestorationDataset(args.train_lr, args.train_gt, patch_size=args.patch_size,
                                          train=True, upsample_lr=args.upsample_lr, augment=True)
    val_full = PairedRestorationDataset(args.train_lr, args.train_gt, patch_size=args.patch_size,
                                        train=False, upsample_lr=args.upsample_lr, augment=False)
    n = len(train_full.pairs)
    train_idx, val_idx = split_indices(n, args.val_frac)
    train_set, val_set = Subset(train_full, train_idx), Subset(val_full, val_idx)
    print(f"total pairs: {n} | train: {len(train_set)} | val: {len(val_set)}")

    histories, models = {}, {}
    for name in ["unet", "nafnet"]:
        model, hist = train_model(name, args, train_set, val_set, device)
        histories[name] = hist
        models[name] = model

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(histories, f, indent=2)
    plot_curves(histories, os.path.join(args.out_dir, "curves.png"))

    panel_idx = list(range(min(args.n_panel, len(val_set))))
    render_comparison_panel(models, val_set, panel_idx,
                            os.path.join(args.out_dir, "comparison_panel.png"), device)

    if args.test_lr:
        render_test_panel(models, args.test_lr, args.n_test_panel,
                          os.path.join(args.out_dir, "test_panel.png"), device, args.upsample_lr)

    print("\n=== final val metrics (best checkpoint) ===")
    for name in ["unet", "nafnet"]:
        p, s = run_validation(models[name], DataLoader(val_set, batch_size=1), device)
        print(f"{name}: PSNR {p:.2f} dB | SSIM {s:.4f}")


if __name__ == "__main__":
    main()
