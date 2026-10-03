"""
07_restore_dic_images.py

Restore real DIC images using the trained NAFNet model.

IMPORTANT:
- Real DIC images normally have no clean ground truth, so PSNR/SSIM/LPIPS
  cannot be computed against a reference image.
- This script therefore reports two no-reference indicators:
    1. Laplacian variance (sharpness proxy; higher = sharper)
    2. Immerkaer noise estimate (lower = estimated noise)
- Always inspect the saved before/after images. A restoration model can
  increase apparent sharpness while also amplifying noise/artifacts.

Expected usage from the SALAAR_KLA_PS01 repository root:

    python 07_restore_dic_images.py ^
        --dic_dir "D:\\path\\to\\DIC_images" ^
        --checkpoint "D:\\path\\to\\nafnet_best.pt" ^
        --out_dir "dic_restore_run"

Optional:
    --upsample_2x
    --limit 20
    --size small

The checkpoint architecture size MUST match the checkpoint used during
training.
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ============================================================
# REPOSITORY IMPORTS
# ============================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(REPO_ROOT, "models")):
    REPO_ROOT = os.path.join(REPO_ROOT, "SALAAR_KLA_PS01")
sys.path.insert(0, REPO_ROOT)

from models.nafnet import build_nafnet
from utils.dataset import _list_images, _load_array


# ============================================================
# IMAGE UTILITIES
# ============================================================

def to_grayscale_float01(arr):
    """
    Convert an arbitrary loaded image/array to float32 grayscale [0,1].

    Supports:
        H x W
        H x W x 1
        H x W x 3
        H x W x 4
    """

    arr = np.asarray(arr)

    # Convert RGB/RGBA to grayscale
    if arr.ndim == 3:

        if arr.shape[-1] == 1:
            arr = arr[..., 0]

        elif arr.shape[-1] >= 3:
            arr = (
                0.299 * arr[..., 0]
                + 0.587 * arr[..., 1]
                + 0.114 * arr[..., 2]
            )

        else:
            raise ValueError(
                f"Unsupported image shape: {arr.shape}"
            )

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D grayscale image, got {arr.shape}"
        )

    arr = arr.astype(np.float32)

    # Handle NaN / Inf
    if not np.isfinite(arr).all():

        finite = np.isfinite(arr)

        if not finite.any():
            raise ValueError(
                "Image contains no finite pixels."
            )

        replacement = np.median(arr[finite])

        arr = np.where(
            finite,
            arr,
            replacement
        )

    amin = float(arr.min())
    amax = float(arr.max())

    # Already [0,1]
    if amin >= 0.0 and amax <= 1.0:
        return np.clip(
            arr,
            0.0,
            1.0
        )

    # uint8-like
    if amax <= 255.0:
        arr = arr / 255.0

    # uint16-like
    elif amax <= 65535.0:
        arr = arr / 65535.0

    # Unusual range
    else:

        if amax > amin:
            arr = (
                arr - amin
            ) / (
                amax - amin
            )

        else:
            arr = np.zeros_like(arr)

    return np.clip(
        arr,
        0.0,
        1.0
    )


# ============================================================
# SHARPNESS METRIC
# ============================================================

def laplacian_variance(img):
    """
    Laplacian variance.

    Higher generally indicates stronger high-frequency structure.

    IMPORTANT:
    Noise and ringing can also increase this number.
    """

    img = np.asarray(
        img,
        dtype=np.float32
    )

    kernel = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float32
    )

    pad = np.pad(
        img,
        1,
        mode="reflect"
    )

    lap = (
        kernel[0, 0] * pad[0:-2, 0:-2]
        + kernel[0, 1] * pad[0:-2, 1:-1]
        + kernel[0, 2] * pad[0:-2, 2:]

        + kernel[1, 0] * pad[1:-1, 0:-2]
        + kernel[1, 1] * pad[1:-1, 1:-1]
        + kernel[1, 2] * pad[1:-1, 2:]

        + kernel[2, 0] * pad[2:, 0:-2]
        + kernel[2, 1] * pad[2:, 1:-1]
        + kernel[2, 2] * pad[2:, 2:]
    )

    return float(
        np.var(lap)
    )


# ============================================================
# NOISE ESTIMATION
# ============================================================

def estimate_noise_sigma(img):
    """
    Immerkaer-style fast single-image noise estimate.

    Lower generally indicates lower estimated high-frequency noise.

    This is a no-reference proxy, NOT a direct measurement of
    physical camera noise.
    """

    img = np.asarray(
        img,
        dtype=np.float32
    )

    h, w = img.shape

    if h < 3 or w < 3:
        return float("nan")

    kernel = np.array(
        [
            [1.0, -2.0, 1.0],
            [-2.0, 4.0, -2.0],
            [1.0, -2.0, 1.0]
        ],
        dtype=np.float32
    )

    pad = np.pad(
        img,
        1,
        mode="reflect"
    )

    conv = (
        kernel[0, 0] * pad[0:-2, 0:-2]
        + kernel[0, 1] * pad[0:-2, 1:-1]
        + kernel[0, 2] * pad[0:-2, 2:]

        + kernel[1, 0] * pad[1:-1, 0:-2]
        + kernel[1, 1] * pad[1:-1, 1:-1]
        + kernel[1, 2] * pad[1:-1, 2:]

        + kernel[2, 0] * pad[2:, 0:-2]
        + kernel[2, 1] * pad[2:, 1:-1]
        + kernel[2, 2] * pad[2:, 2:]
    )

    sigma = (
        np.sqrt(np.pi / 2.0)
        * np.sum(np.abs(conv))
        / (
            6.0
            * (w - 2)
            * (h - 2)
        )
    )

    return float(sigma)


# ============================================================
# DISPLAY RANGE
# ============================================================

def percentile_range(
    img,
    low=1.0,
    high=99.0
):
    """
    Robust display range.

    Same range is used for input and restored image so the
    visual comparison is meaningful.
    """

    lo, hi = np.percentile(
        img,
        [
            low,
            high
        ]
    )

    if hi <= lo:

        lo = float(img.min())
        hi = float(img.max())

    if hi <= lo:
        hi = lo + 1e-6

    return float(lo), float(hi)


# ============================================================
# PADDING
# ============================================================

def pad_to_multiple(
    x,
    multiple=8
):
    """
    Pad H/W to a multiple of `multiple`.

    Returns:
        padded tensor
        original H
        original W
    """

    _, _, h, w = x.shape

    pad_h = (
        multiple
        - h % multiple
    ) % multiple

    pad_w = (
        multiple
        - w % multiple
    ) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, h, w

    # Reflection padding for normal image sizes
    if h > pad_h and w > pad_w:

        x = F.pad(
            x,
            (
                0,
                pad_w,
                0,
                pad_h
            ),
            mode="reflect"
        )

    else:

        # Safer for very small images
        x = F.pad(
            x,
            (
                0,
                pad_w,
                0,
                pad_h
            ),
            mode="replicate"
        )

    return x, h, w


# ============================================================
# RESTORATION
# ============================================================

@torch.inference_mode()
def restore_one(
    model,
    arr,
    device,
    pad_multiple=8
):
    """
    Restore one H x W image.

    Input:
        float32 [0,1]

    Output:
        float32 [0,1]
    """

    x = (
        torch.from_numpy(arr)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    x, h, w = pad_to_multiple(
        x,
        pad_multiple
    )

    # CUDA inference can use FP16
    if device.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            y = model(x)

    else:
        y = model(x)

    # Remove padding
    y = y[
        ...,
        :h,
        :w
    ]

    # Keep valid image range
    y = y.clamp(
        0.0,
        1.0
    )

    return (
        y
        .squeeze()
        .float()
        .cpu()
        .numpy()
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint,
    device
):
    """
    Supports:

        {'model': state_dict, ...}

    or:

        {'state_dict': state_dict, ...}

    or:

        raw state_dict
    """

    checkpoint_obj = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False
    )

    if isinstance(
        checkpoint_obj,
        dict
    ):

        if "model" in checkpoint_obj:

            state_dict = checkpoint_obj["model"]

        elif "state_dict" in checkpoint_obj:

            state_dict = checkpoint_obj["state_dict"]

        else:

            state_dict = checkpoint_obj

    else:

        state_dict = checkpoint_obj

    # Remove DataParallel prefix
    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):
            key = key[
                len("module."):]
        
        cleaned[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned,
            strict=False
        )
    )

    if missing:

        raise RuntimeError(
            "Checkpoint is missing model parameters.\n"
            f"First missing keys: {missing[:10]}"
        )

    if unexpected:

        print(
            "Warning: checkpoint contains "
            f"{len(unexpected)} unexpected parameters."
        )

        print(
            f"First few: {unexpected[:5]}"
        )

    return checkpoint_obj


# ============================================================
# SAVE COMPARISON IMAGE
# ============================================================

def save_comparison(
    input_img,
    restored_img,
    input_name,
    output_path,
    title_left="DIC input",
    title_right="NAFNet restored"
):
    """
    Save:

        INPUT | RESTORED

    using the SAME display range.
    """

    vmin, vmax = percentile_range(
        input_img
    )

    sharp_in = laplacian_variance(
        input_img
    )

    sharp_out = laplacian_variance(
        restored_img
    )

    noise_in = estimate_noise_sigma(
        input_img
    )

    noise_out = estimate_noise_sigma(
        restored_img
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True
    )

    axes[0].imshow(
        input_img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    axes[0].set_title(
        title_left,
        fontsize=13
    )

    axes[1].imshow(
        restored_img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    axes[1].set_title(
        title_right,
        fontsize=13
    )

    axes[0].set_xlabel(
        f"Laplacian variance: {sharp_in:.2f}\n"
        f"Noise estimate σ: {noise_in:.4f}",
        fontsize=9
    )

    axes[1].set_xlabel(
        f"Laplacian variance: {sharp_out:.2f}\n"
        f"Noise estimate σ: {noise_out:.4f}",
        fontsize=9
    )

    for ax in axes:

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        input_name,
        fontsize=14
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    rows,
    output_path,
    checkpoint,
    device,
    upsample_2x
):

    if not rows:
        return

    def mean(key):

        values = [
            r[key]
            for r in rows
            if np.isfinite(
                r[key]
            )
        ]

        if not values:
            return float("nan")

        return float(
            np.mean(values)
        )

    sharp_in = mean(
        "sharpness_input"
    )

    sharp_out = mean(
        "sharpness_restored"
    )

    noise_in = mean(
        "noise_input"
    )

    noise_out = mean(
        "noise_restored"
    )

    sharp_change = (
        100.0
        * (
            sharp_out
            - sharp_in
        )
        / max(
            abs(sharp_in),
            1e-12
        )
    )

    noise_change = (
        100.0
        * (
            noise_out
            - noise_in
        )
        / max(
            abs(noise_in),
            1e-12
        )
    )

    text = f"""
DIC IMAGE RESTORATION — NO-REFERENCE SUMMARY
=============================================

Number of images: {len(rows)}

Checkpoint:
{checkpoint}

Device:
{device}

2x bicubic pre-upsample:
{upsample_2x}


IMPORTANT
---------

There is no clean ground-truth image for these real DIC images.

Therefore:

    PSNR
    SSIM
    LPIPS

are NOT reported here.

The following no-reference indicators are reported instead.


1. LAPLACIAN VARIANCE
---------------------

Higher values generally indicate stronger high-frequency structure.

However, noise and ringing can ALSO increase this value.


Input average:
    {sharp_in:.4f}

Restored average:
    {sharp_out:.4f}

Change:
    {sharp_change:+.2f}%


2. NOISE ESTIMATE
-----------------

Immerkaer-style noise estimate.

Lower values generally indicate lower estimated high-frequency noise.

Input average:
    {noise_in:.6f}

Restored average:
    {noise_out:.6f}

Change:
    {noise_change:+.2f}%


INTERPRETATION
--------------

Do NOT judge restoration from either metric alone.

Inspect the saved comparison images for:

    - preservation of DIC speckle texture
    - suppression of unwanted noise
    - preservation of physical edges
    - absence of hallucinated structures
    - absence of ringing / oversharpening
    - preservation of features relevant to DIC tracking


EXPERIMENTAL STATUS
-------------------

This should be described as an OUT-OF-DISTRIBUTION
generalization test unless clean DIC ground truth is available.

The KLA training data and the real DIC images are not necessarily
subject to identical degradation processes.
"""

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            text.strip()
            + "\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Restore real DIC images using "
            "a trained NAFNet checkpoint."
        )
    )

    parser.add_argument(
        "--dic_dir",
        required=True,
        help=(
            "Folder containing real DIC images."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to trained NAFNet checkpoint."
        )
    )

    parser.add_argument(
        "--out_dir",
        default="dic_restore_run",
        help=(
            "Output directory."
        )
    )

    parser.add_argument(
        "--size",
        choices=[
            "tiny",
            "small",
            "base"
        ],
        default="small",
        help=(
            "NAFNet size. MUST match "
            "checkpoint architecture."
        )
    )

    parser.add_argument(
        "--upsample_2x",
        action="store_true",
        help=(
            "Bicubic-upsample input 2x before restoration. "
            "Use ONLY if the real DIC image represents the "
            "same downsampled-resolution condition used "
            "during training."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N images."
        )
    )

    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=8,
        help=(
            "Pad H/W to a multiple of this value."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    dic_dir = os.path.abspath(
        args.dic_dir
    )

    checkpoint = os.path.abspath(
        args.checkpoint
    )

    out_dir = os.path.abspath(
        args.out_dir
    )

    if not os.path.isdir(
        dic_dir
    ):

        raise FileNotFoundError(
            f"DIC image directory does not exist:\n"
            f"{dic_dir}"
        )

    if not os.path.isfile(
        checkpoint
    ):

        raise FileNotFoundError(
            f"Checkpoint does not exist:\n"
            f"{checkpoint}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DIC IMAGE RESTORATION"
    )

    print(
        "=" * 72
    )

    print(
        f"DIC directory : {dic_dir}"
    )

    print(
        f"Checkpoint    : {checkpoint}"
    )

    print(
        f"Output        : {out_dir}"
    )

    print(
        f"Device        : {device}"
    )

    print(
        f"NAFNet size   : {args.size}"
    )

    print(
        f"2x upsample   : {args.upsample_2x}"
    )

    print(
        "=" * 72
    )

    if device.type == "cuda":

        print(
            "GPU           : "
            + torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    comparison_dir = os.path.join(
        out_dir,
        "comparison"
    )

    restored_dir = os.path.join(
        out_dir,
        "restored"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    os.makedirs(
        comparison_dir,
        exist_ok=True
    )

    os.makedirs(
        restored_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_nafnet(
        in_channels=1,
        out_channels=1,
        size=args.size
    ).to(device)

    checkpoint_obj = load_checkpoint(
        model,
        checkpoint,
        device
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # Print useful metadata if present
    if isinstance(
        checkpoint_obj,
        dict
    ):

        for key in [
            "epoch",
            "best_metric",
            "val_psnr",
            "val_ssim",
            "val_lpips"
        ]:

            if key in checkpoint_obj:

                print(
                    f"Checkpoint {key}: "
                    f"{checkpoint_obj[key]}"
                )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    paths = _list_images(
        dic_dir
    )

    if args.limit is not None:

        if args.limit <= 0:

            raise ValueError(
                "--limit must be positive."
            )

        paths = paths[
            :args.limit
        ]

    if not paths:

        raise FileNotFoundError(
            f"No supported images found in:\n"
            f"{dic_dir}"
        )

    print(
        f"Images found  : {len(paths)}"
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    rows = []

    for index, path in enumerate(
        paths,
        start=1
    ):

        filename = os.path.basename(
            path
        )

        stem = os.path.splitext(
            filename
        )[0]

        try:

            # ------------------------------------------------
            # Load
            # ------------------------------------------------

            raw = _load_array(
                path
            )

            input_img = (
                to_grayscale_float01(
                    raw
                )
            )

            # ------------------------------------------------
            # Optional 2x upsampling
            # ------------------------------------------------

            model_input = input_img

            if args.upsample_2x:

                h, w = (
                    model_input.shape
                )

                pil = Image.fromarray(
                    np.clip(
                        model_input * 255.0,
                        0,
                        255
                    ).astype(
                        np.uint8
                    )
                )

                pil = pil.resize(
                    (
                        w * 2,
                        h * 2
                    ),
                    Image.Resampling.BICUBIC
                )

                model_input = (
                    np.asarray(
                        pil,
                        dtype=np.float32
                    )
                    / 255.0
                )

            # ------------------------------------------------
            # Restore
            # ------------------------------------------------

            restored = restore_one(
                model,
                model_input,
                device,
                pad_multiple=args.pad_multiple
            )

            # ------------------------------------------------
            # No-reference metrics
            # ------------------------------------------------

            sharp_in = (
                laplacian_variance(
                    model_input
                )
            )

            sharp_out = (
                laplacian_variance(
                    restored
                )
            )

            noise_in = (
                estimate_noise_sigma(
                    model_input
                )
            )

            noise_out = (
                estimate_noise_sigma(
                    restored
                )
            )

            sharp_change = (
                100.0
                * (
                    sharp_out
                    - sharp_in
                )
                / max(
                    abs(sharp_in),
                    1e-12
                )
            )

            noise_change = (
                100.0
                * (
                    noise_out
                    - noise_in
                )
                / max(
                    abs(noise_in),
                    1e-12
                )
            )

            # ------------------------------------------------
            # Store metrics
            # ------------------------------------------------

            rows.append(
                {
                    "image": filename,

                    "height":
                        int(
                            model_input.shape[0]
                        ),

                    "width":
                        int(
                            model_input.shape[1]
                        ),

                    "sharpness_input":
                        float(
                            sharp_in
                        ),

                    "sharpness_restored":
                        float(
                            sharp_out
                        ),

                    "sharpness_change_pct":
                        float(
                            sharp_change
                        ),

                    "noise_input":
                        float(
                            noise_in
                        ),

                    "noise_restored":
                        float(
                            noise_out
                        ),

                    "noise_change_pct":
                        float(
                            noise_change
                        )
                }
            )

            # ------------------------------------------------
            # Save restored image
            # ------------------------------------------------

            restored_uint8 = np.clip(
                restored * 255.0,
                0,
                255
            ).astype(
                np.uint8
            )

            restored_path = os.path.join(
                restored_dir,
                f"{stem}_restored.png"
            )

            Image.fromarray(
                restored_uint8,
                mode="L"
            ).save(
                restored_path
            )

            # ------------------------------------------------
            # Save comparison
            # ------------------------------------------------

            comparison_path = os.path.join(
                comparison_dir,
                f"{stem}_compare.png"
            )

            save_comparison(
                model_input,
                restored,
                filename,
                comparison_path
            )

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"[{index:4d}/{len(paths):4d}] "
                f"{filename:40s} | "
                f"sharp "
                f"{sharp_in:10.3f} -> "
                f"{sharp_out:10.3f} "
                f"({sharp_change:+7.2f}%) | "
                f"noise "
                f"{noise_in:.5f} -> "
                f"{noise_out:.5f} "
                f"({noise_change:+7.2f}%)"
            )

        except Exception as exc:

            print(
                f"[{index:4d}/{len(paths):4d}] "
                f"FAILED: {filename} | "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Ensure at least one success
    # --------------------------------------------------------

    if not rows:

        raise RuntimeError(
            "Every DIC image failed during processing. "
            "Check image format and checkpoint architecture."
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        out_dir,
        "quality_metrics.csv"
    )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        out_dir,
        "summary.txt"
    )

    write_summary(
        rows,
        summary_path,
        checkpoint,
        device,
        args.upsample_2x
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DIC RESTORATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Processed successfully : "
        f"{len(rows)} / {len(paths)}"
    )

    print(
        f"Restored images        : "
        f"{restored_dir}"
    )

    print(
        f"Comparisons            : "
        f"{comparison_dir}"
    )

    print(
        f"Metrics CSV            : "
        f"{csv_path}"
    )

    print(
        f"Summary                : "
        f"{summary_path}"
    )

    print(
        "=" * 72
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()