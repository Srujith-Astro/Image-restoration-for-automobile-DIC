"""
07_restore_dic_images.py -- restore real DIC images with the trained NAFNet
checkpoint and build side-by-side comparison panels.

IMPORTANT ON METRICS: PSNR, SSIM, and LPIPS are all *full-reference* metrics
-- they measure distance to a known-clean ground truth. Real DIC camera
images have no such reference (that's the whole point of restoring them),
so none of those three can be computed honestly here. Reporting a PSNR/SSIM
number against nothing, or against the noisy image itself, would be
meaningless and should not go on the poster.

What this script gives you instead: three *no-reference* quality proxies,
computed for both the noisy input and the restored output, so you can show
a real, defensible before/after number even with no ground truth:
  - sharpness   = variance of the Laplacian (higher = sharper / more detail)
  - noise_sigma = Immerkaer (1996) fast noise-level estimate (lower = cleaner)
  - hf_fraction = fraction of FFT energy in the high-frequency band
                  (interpretation depends on the image -- rising HF energy
                  after restoration can mean recovered detail OR
                  reintroduced noise; always eyeball the panel too)

These are the same families of feature already used in your project's own
dataset-forensics stage (edge/Laplacian, FFT energy, residual statistics),
so the vocabulary matches your report.

For an actual quantitative validation (real PSNR/SSIM/LPIPS numbers), the
honest route is: take any DIC image you'd call clean, synthetically degrade
it with the same pipeline used for training (downsample + speckle + Gaussian
noise), restore it, and compare against the real clean original. That gives
true full-reference numbers at the cost of no longer being a real camera
capture. Ask if you want a script for that -- it's a different, smaller tool.

Usage:
    python 07_restore_dic_images.py --dic_dir "path/to/DIC_images" ^
        --checkpoint nafnet_best.pt --out_dir dic_restored

Output in --out_dir:
    panels/<image>_panel.png   noisy | restored, proxy numbers annotated
    restored/<image>.png        restored image alone, full resolution
    quality_report.csv          per-image proxy numbers (noisy vs restored)
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from models.nafnet import build_nafnet
from models.unet import build_unet
from utils.dataset import _list_images, _load_array

LAPLACIAN_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
# Immerkaer (1996) fast noise estimator kernel
NOISE_KERNEL = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])


def sharpness(arr):
    k = LAPLACIAN_KERNEL.view(1, 1, 3, 3)
    x = torch.from_numpy(arr).float()[None, None]
    lap = F.conv2d(x, k, padding=1)
    return lap.var().item()


def noise_sigma(arr):
    h, w = arr.shape
    k = NOISE_KERNEL.view(1, 1, 3, 3)
    x = torch.from_numpy(arr).float()[None, None]
    conv = F.conv2d(x, k, padding=1)
    sigma = conv.abs().sum().item() * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2) + 1e-8)
    return sigma


def hf_fraction(arr, cutoff=0.25):
    f = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(f)
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    hf_mask = r > cutoff
    total = mag.sum() + 1e-8
    return float(mag[hf_mask].sum() / total)


def bicubic_upsample(arr, factor):
    if factor == 1:
        return arr
    h, w = arr.shape
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img = img.resize((w * factor, h * factor), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dic_dir", required=True, help="folder of real DIC images")
    ap.add_argument("--checkpoint", default="nafnet_best.pt")
    ap.add_argument("--model", choices=["nafnet", "unet"], default="nafnet")
    ap.add_argument("--out_dir", default="dic_restored")
    ap.add_argument("--upsample_factor", type=int, default=2,
                     help="match whatever factor the checkpoint was trained on (2 by default)")
    ap.add_argument("--assume_already_upsampled", action="store_true",
                     help="pass this if your DIC images are already at the target resolution "
                          "(skip bicubic upsampling, feed them to the model as-is)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    fn = build_nafnet if args.model == "nafnet" else build_unet
    model = fn(in_channels=1, out_channels=1, size="small").to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"Loaded {args.model} checkpoint: {args.checkpoint}")

    paths = _list_images(args.dic_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.dic_dir}")
    print(f"Found {len(paths)} DIC images")

    panel_dir = os.path.join(args.out_dir, "panels")
    restored_dir = os.path.join(args.out_dir, "restored")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(restored_dir, exist_ok=True)

    rows = []
    factor = 1 if args.assume_already_upsampled else args.upsample_factor
    for p in paths:
        name = os.path.basename(p)
        arr = _load_array(p)
        noisy = bicubic_upsample(arr, factor)

        x = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
        restored = model(x).clamp(0, 1).squeeze().cpu().numpy()

        Image.fromarray((restored * 255).astype(np.uint8)).save(
            os.path.join(restored_dir, name))

        m_noisy = dict(sharpness=sharpness(noisy), noise_sigma=noise_sigma(noisy),
                       hf_fraction=hf_fraction(noisy))
        m_rest = dict(sharpness=sharpness(restored), noise_sigma=noise_sigma(restored),
                      hf_fraction=hf_fraction(restored))

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
        axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Noisy (input)")
        axes[0].set_xlabel(f"sharpness {m_noisy['sharpness']:.4f}\n"
                           f"noise_sigma {m_noisy['noise_sigma']:.4f}\n"
                           f"HF fraction {m_noisy['hf_fraction']:.4f}", fontsize=8)
        axes[1].imshow(restored, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"{args.model} restored")
        axes[1].set_xlabel(f"sharpness {m_rest['sharpness']:.4f}\n"
                           f"noise_sigma {m_rest['noise_sigma']:.4f}\n"
                           f"HF fraction {m_rest['hf_fraction']:.4f}", fontsize=8)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(name, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(panel_dir, os.path.splitext(name)[0] + "_panel.png"), dpi=150)
        plt.close()

        rows.append(dict(image=name,
                         noisy_sharpness=m_noisy["sharpness"], restored_sharpness=m_rest["sharpness"],
                         noisy_noise_sigma=m_noisy["noise_sigma"], restored_noise_sigma=m_rest["noise_sigma"],
                         noisy_hf_fraction=m_noisy["hf_fraction"], restored_hf_fraction=m_rest["hf_fraction"]))
        print(f"{name}: sharpness {m_noisy['sharpness']:.4f} -> {m_rest['sharpness']:.4f} | "
              f"noise_sigma {m_noisy['noise_sigma']:.4f} -> {m_rest['noise_sigma']:.4f}")

    with open(os.path.join(args.out_dir, "quality_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    print("\n=== averages across all DIC images ===")
    print(f"sharpness   : noisy {avg('noisy_sharpness'):.4f} -> restored {avg('restored_sharpness'):.4f}")
    print(f"noise_sigma : noisy {avg('noisy_noise_sigma'):.4f} -> restored {avg('restored_noise_sigma'):.4f}")
    print(f"hf_fraction : noisy {avg('noisy_hf_fraction'):.4f} -> restored {avg('restored_hf_fraction'):.4f}")
    print(f"\nSaved panels to {panel_dir}")
    print(f"Saved restored images to {restored_dir}")
    print(f"Saved {os.path.join(args.out_dir, 'quality_report.csv')}")


if __name__ == "__main__":
    main()
