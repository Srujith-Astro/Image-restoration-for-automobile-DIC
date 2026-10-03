"""
07_restore_dic_images.py -- restore real DIC images with a trained checkpoint
and build side-by-side comparison panels, SWEEPING over a few candidate
upsample factors since we don't know how (or whether) these DIC images were
actually downsampled relative to what the model was trained on.

IMPORTANT ON METRICS: PSNR, SSIM, and LPIPS are all *full-reference* metrics
-- they measure distance to a known-clean ground truth. Real DIC camera
images have no such reference, so none of those three can be computed
honestly here. Instead this script reports three *no-reference* proxies,
computed for the noisy input and for each restored candidate:
  - sharpness   = variance of the Laplacian (higher = sharper / more detail)
  - noise_sigma = Immerkaer (1996) fast noise-level estimate (lower = cleaner)
  - hf_fraction = fraction of FFT energy in the high-frequency band
A GOOD restoration should raise sharpness and/or lower noise_sigma relative
to the noisy input. If every factor you try instead LOWERS sharpness, that's
a real signal the model is just blurring -- not evidence you picked the
wrong factor.

Usage (sweeps factor=1 and factor=2 by default; add more if you want):
    python 07_restore_dic_images.py --dic_dir "path/to/DIC_images" ^
        --checkpoint nafnet_best.pt --out_dir dic_restored --upsample_factors 1,2

Output in --out_dir:
    panels/<image>_panel.png    noisy | restored@f1 | restored@f2 | ...
    restored/<image>_f<factor>.png
    quality_report.csv           one row per (image, factor)
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
if not os.path.isdir(os.path.join(REPO_ROOT, "models")):
    REPO_ROOT = os.path.join(REPO_ROOT, "SALAAR_KLA_PS01")
sys.path.insert(0, REPO_ROOT)
from models.nafnet import build_nafnet
from models.unet import build_unet
from utils.dataset import _list_images, _load_array

LAPLACIAN_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
NOISE_KERNEL = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])  # Immerkaer (1996)


def sharpness(arr):
    k = LAPLACIAN_KERNEL.view(1, 1, 3, 3)
    x = torch.from_numpy(arr).float()[None, None]
    return F.conv2d(x, k, padding=1).var().item()


def noise_sigma(arr):
    h, w = arr.shape
    k = NOISE_KERNEL.view(1, 1, 3, 3)
    x = torch.from_numpy(arr).float()[None, None]
    conv = F.conv2d(x, k, padding=1)
    return conv.abs().sum().item() * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2) + 1e-8)


def hf_fraction(arr, cutoff=0.25):
    f = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(f)
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    total = mag.sum() + 1e-8
    return float(mag[r > cutoff].sum() / total)


def quality(arr):
    return dict(sharpness=sharpness(arr), noise_sigma=noise_sigma(arr), hf_fraction=hf_fraction(arr))


def resample(arr, factor):
    """factor > 1: upsample by that integer factor. factor < 1 (e.g. 0.5): first
    DOWNsample by 1/factor then upsample back by the same amount -- simulates
    'this image is already high-res, degrade it the way training data was
    degraded, then feed it through' in case your DIC images were never
    downsampled at all relative to the model's expectations."""
    h, w = arr.shape
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if factor == 1:
        return arr
    if factor > 1:
        img = img.resize((int(round(w * factor)), int(round(h * factor))), Image.BICUBIC)
    else:
        inv = int(round(1 / factor))
        img = img.resize((w // inv, h // inv), Image.BICUBIC)
        img = img.resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dic_dir", required=True, help="folder of real DIC images")
    ap.add_argument("--checkpoint", default="nafnet_best.pt")
    ap.add_argument("--model", choices=["nafnet", "unet"], default="nafnet")
    ap.add_argument("--out_dir", default="dic_restored")
    ap.add_argument("--upsample_factors", default="1,2",
                     help="comma-separated factors to try, e.g. '1,2' or '0.5,1,2'. "
                          "1 = feed image as-is (no resampling). 2 = bicubic-upsample 2x "
                          "(what the released checkpoint was trained on). A factor below 1 "
                          "(e.g. 0.5) first downsamples then upsamples back, simulating the "
                          "training degradation in case your images were never downsampled.")
    args = ap.parse_args()

    factors = [float(f) for f in args.upsample_factors.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | factors to test: {factors}")

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
    for p in paths:
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        arr = _load_array(p)

        ncols = 1 + len(factors)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.6))

        base_q = quality(arr)
        axes[0].imshow(arr, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original (as given)")
        axes[0].set_xlabel(f"sharpness {base_q['sharpness']:.4f}\n"
                           f"noise_sigma {base_q['noise_sigma']:.4f}\n"
                           f"HF fraction {base_q['hf_fraction']:.4f}", fontsize=8)
        rows.append(dict(image=name, factor="input_only", **base_q))

        for i, factor in enumerate(factors, start=1):
            noisy = resample(arr, factor)
            x = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
            restored = model(x).clamp(0, 1).squeeze().cpu().numpy()

            Image.fromarray((restored * 255).astype(np.uint8)).save(
                os.path.join(restored_dir, f"{stem}_f{factor}.png"))

            q = quality(restored)
            axes[i].imshow(restored, cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(f"restored, factor={factor}")
            axes[i].set_xlabel(f"sharpness {q['sharpness']:.4f}\n"
                               f"noise_sigma {q['noise_sigma']:.4f}\n"
                               f"HF fraction {q['hf_fraction']:.4f}", fontsize=8)
            rows.append(dict(image=name, factor=factor, **q))

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(name, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(panel_dir, f"{stem}_panel.png"), dpi=150)
        plt.close()

        print(f"{name}: input sharpness {base_q['sharpness']:.4f} | "
              + " | ".join(f"f={f} -> {q['sharpness']:.4f}" for f, q in
                           zip(factors, [r for r in rows if r['image'] == name and r['factor'] != 'input_only'])))

    with open(os.path.join(args.out_dir, "quality_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    print("\n=== average sharpness by factor (higher = less blurring) ===")
    for tag in ["input_only"] + factors:
        vals = [r["sharpness"] for r in rows if r["factor"] == tag]
        if vals:
            print(f"  factor={tag:<12} avg sharpness {sum(vals)/len(vals):.5f}")
    print("\n=== average noise_sigma by factor (lower = cleaner) ===")
    for tag in ["input_only"] + factors:
        vals = [r["noise_sigma"] for r in rows if r["factor"] == tag]
        if vals:
            print(f"  factor={tag:<12} avg noise_sigma {sum(vals)/len(vals):.5f}")

    print(f"\nSaved panels to {panel_dir}")
    print(f"Saved restored images to {restored_dir}")
    print(f"Saved {os.path.join(args.out_dir, 'quality_report.csv')}")
    print("\nLook for the factor whose restored sharpness is CLOSEST TO OR ABOVE the "
          "'input_only' row, and whose noise_sigma is clearly lower. If every factor's "
          "restored sharpness is well below input_only, the model is blurring regardless "
          "of factor -- that's a domain-gap finding, not a factor-tuning problem.")


if __name__ == "__main__":
    main()
