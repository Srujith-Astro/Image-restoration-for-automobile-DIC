"""
AUTOMOTIVE IMAGE RESTORATION EXPERIMENT
========================================

Pipeline:

Clean automotive images
        |
        v
Generate controlled degradation
        |
        v
Save degraded + GT pairs
        |
        v
Train U-Net / NAFNet / Restormer
        |
        v
Restore degraded test images
        |
        v
PSNR / SSIM / LPIPS
        |
        v
Visual comparison

FIRST EXPERIMENT:
    2x spatial downsampling

GT:
    256 x 256

Degraded:
    128 x 128 -> bicubic upsample -> 256 x 256

Models:
    U-Net
    NAFNet
    Restormer
"""

import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import structural_similarity
import pandas as pd


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training, but no CUDA-enabled GPU is available."
        )

    try:
        torch.cuda.set_device(0)
        torch.empty(1, device="cuda:0")
        torch.cuda.synchronize(0)
    except RuntimeError as exc:
        raise RuntimeError(
            "CUDA is required for training, but the GPU failed its startup check."
        ) from exc

    return torch.device("cuda:0")


DEVICE = require_cuda()
AMP_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)
USE_GRAD_SCALER = AMP_DTYPE == torch.float16


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS
CLEAN_DIR = Path(r"F:\Hack\kla hack 3\high res car defect")

# Everything else is generated automatically
ROOT = Path(r"F:\Hack\kla hack 3\automotive_image_restoration")

GT_DIR = ROOT / "ground_truth"
DEGRADED_DIR = ROOT / "degraded"

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
COMPARISON_DIR = ROOT / "comparisons"
METRICS_DIR = ROOT / "metrics"


# -----------------------------
# Dataset
# -----------------------------

IMAGE_SIZE = 256

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

SEED = 42


# -----------------------------
# Degradation
# -----------------------------

DOWN_FACTOR = 2


# -----------------------------
# Training
# -----------------------------

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 6

NUM_WORKERS = 4


# -----------------------------
# Models
# -----------------------------

MODELS_TO_TRAIN = [
    "restormer",
]


# ============================================================
# DIRECTORIES
# ============================================================

for d in [
    GT_DIR,
    DEGRADED_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    COMPARISON_DIR,
    METRICS_DIR,
]:

    d.mkdir(
        parents=True,
        exist_ok=True
    )


for model_name in MODELS_TO_TRAIN:

    (OUTPUT_DIR / model_name).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

print("=" * 80)
print("AUTOMOTIVE IMAGE RESTORATION")
print("=" * 80)

print("Device:", DEVICE)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(
            0
        ).total_memory / 2**30,
        2
    ),
    "GB"
)


# ============================================================
# IMAGE UTILITIES
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_images(directory):

    files = []

    for p in directory.rglob("*"):

        if p.suffix.lower() in VALID_EXTENSIONS:

            files.append(p)

    return sorted(files)


def load_image(path):

    image = Image.open(path).convert("RGB")

    return image


def center_crop_or_resize(image, size):

    image = image.copy()

    w, h = image.size

    # Resize while preserving aspect ratio
    scale = max(
        size / w,
        size / h
    )

    new_w = int(w * scale)

    new_h = int(h * scale)

    image = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # Center crop
    left = (new_w - size) // 2

    top = (new_h - size) // 2

    right = left + size

    bottom = top + size

    image = image.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    return image


# ============================================================
# DEGRADATION
# ============================================================

def degrade_image(image):

    """
    Main experiment:

        256x256 GT
             |
             v
        128x128
             |
             v
        256x256 bicubic

    The final degraded image therefore has the
    same dimensions as GT, making supervised
    restoration straightforward.
    """

    image = image.resize(
        (
            IMAGE_SIZE // DOWN_FACTOR,
            IMAGE_SIZE // DOWN_FACTOR
        ),
        Image.Resampling.BICUBIC
    )

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BICUBIC
    )

    return image


# ============================================================
# CREATE DATASET
# ============================================================

def prepare_dataset():

    print("\n" + "=" * 80)

    print("STEP 1: PREPARING DATASET")

    print("=" * 80)

    images = get_images(CLEAN_DIR)

    print(
        "Found",
        len(images),
        "clean images"
    )

    if len(images) == 0:

        raise RuntimeError(
            f"No images found in {CLEAN_DIR}"
        )

    random.shuffle(images)

    n = len(images)

    n_train = int(
        n * TRAIN_FRACTION
    )

    n_val = int(
        n * VAL_FRACTION
    )

    train_images = images[
        :n_train
    ]

    val_images = images[
        n_train:n_train + n_val
    ]

    test_images = images[
        n_train + n_val:
    ]

    splits = {

        "train": train_images,

        "val": val_images,

        "test": test_images,

    }

    print("\nSplit:")

    for name, imgs in splits.items():

        print(
            f"{name:>5}:",
            len(imgs)
        )

    # Store split lists
    for split_name, split_images in splits.items():

        gt_split_dir = (
            GT_DIR / split_name
        )

        deg_split_dir = (
            DEGRADED_DIR / split_name
        )

        gt_split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        deg_split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for idx, source_path in enumerate(
            split_images
        ):

            image = load_image(
                source_path
            )

            image = center_crop_or_resize(
                image,
                IMAGE_SIZE
            )

            degraded = degrade_image(
                image
            )

            filename = (
                f"{idx:06d}.png"
            )

            image.save(
                gt_split_dir / filename
            )

            degraded.save(
                deg_split_dir / filename
            )

    print("\nDataset created.")

    print(
        "GT:",
        GT_DIR
    )

    print(
        "Degraded:",
        DEGRADED_DIR
    )


# ============================================================
# DATASET CLASS
# ============================================================

class RestorationDataset(
    Dataset
):

    def __init__(
        self,
        degraded_dir,
        gt_dir,
        augment=False
    ):

        self.degraded_dir = Path(
            degraded_dir
        )

        self.gt_dir = Path(
            gt_dir
        )

        self.augment = augment

        self.files = sorted(
            [
                p
                for p in self.degraded_dir.iterdir()
                if p.suffix.lower() == ".png"
            ]
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        degraded_path = self.files[idx]

        gt_path = (
            self.gt_dir /
            degraded_path.name
        )

        degraded = np.asarray(
            Image.open(
                degraded_path
            ).convert("RGB")
        ).astype(
            np.float32
        ) / 255.0

        gt = np.asarray(
            Image.open(
                gt_path
            ).convert("RGB")
        ).astype(
            np.float32
        ) / 255.0

        # --------------------------------
        # augmentation
        # --------------------------------

        if self.augment:

            if random.random() < 0.5:

                degraded = np.flip(
                    degraded,
                    axis=1
                ).copy()

                gt = np.flip(
                    gt,
                    axis=1
                ).copy()

            if random.random() < 0.5:

                degraded = np.flip(
                    degraded,
                    axis=0
                ).copy()

                gt = np.flip(
                    gt,
                    axis=0
                ).copy()

            k = random.randint(
                0,
                3
            )

            degraded = np.rot90(
                degraded,
                k
            ).copy()

            gt = np.rot90(
                gt,
                k
            ).copy()

        degraded = torch.from_numpy(
            degraded
        ).permute(
            2,
            0,
            1
        ).float()

        gt = torch.from_numpy(
            gt
        ).permute(
            2,
            0,
            1
        ).float()

        return degraded, gt, degraded_path.name


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True),

        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(
            3,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            3,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return torch.sigmoid(
            self.final(d1)
        )


# ============================================================
# NAFNET
# ============================================================

class NAFBlock(nn.Module):

    def __init__(
        self,
        channels,
        dropout=0.0
    ):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            1,
            channels
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels * 2,
            1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels
        )

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.norm2 = nn.GroupNorm(
            1,
            channels
        )

        self.ffn1 = nn.Conv2d(
            channels,
            channels * 2,
            1
        )

        self.ffn2 = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

    def forward(self, x):

        y = self.norm1(x)

        y = self.conv1(y)

        y1, y2 = y.chunk(
            2,
            dim=1
        )

        y = y1 * y2

        y = self.conv2(y)

        y = self.conv3(y)

        y = self.dropout1(y)

        x = x + y * self.beta

        y = self.norm2(x)

        y = self.ffn1(y)

        y1, y2 = y.chunk(
            2,
            dim=1
        )

        y = y1 * y2

        y = self.ffn2(y)

        y = self.dropout2(y)

        return x + y * self.gamma


class NAFNet(nn.Module):

    def __init__(
        self,
        width=32,
        blocks=8
    ):

        super().__init__()

        self.intro = nn.Conv2d(
            3,
            width,
            3,
            padding=1
        )

        self.blocks = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(blocks)
            ]
        )

        self.ending = nn.Conv2d(
            width,
            3,
            3,
            padding=1
        )

    def forward(self, x):

        residual = x

        x = self.intro(x)

        x = self.blocks(x)

        x = self.ending(x)

        return torch.clamp(
            x + residual,
            0,
            1
        )


# ============================================================
# RESTORMER-LITE
# ============================================================

class MDTA(nn.Module):

    def __init__(
        self,
        dim,
        heads=4
    ):

        super().__init__()

        self.heads = heads

        self.temperature = nn.Parameter(
            torch.ones(heads, 1, 1)
        )

        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            1,
            bias=False
        )

        self.qkv_dw = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            padding=1,
            groups=dim * 3,
            bias=False
        )

        self.project = nn.Conv2d(
            dim,
            dim,
            1
        )

    def forward(self, x):

        b, c, h, w = x.shape

        qkv = self.qkv_dw(
            self.qkv(x)
        )

        q, k, v = qkv.chunk(
            3,
            dim=1
        )

        q = q.reshape(
            b,
            self.heads,
            c // self.heads,
            h * w
        )

        k = k.reshape(
            b,
            self.heads,
            c // self.heads,
            h * w
        )

        v = v.reshape(
            b,
            self.heads,
            c // self.heads,
            h * w
        )

        q = F.normalize(
            q,
            dim=-1
        )

        k = F.normalize(
            k,
            dim=-1
        )

        attn = (
            q @ k.transpose(
                -2,
                -1
            )
        ) * self.temperature

        attn = attn.softmax(
            dim=-1
        )

        out = attn @ v

        out = out.reshape(
            b,
            c,
            h,
            w
        )

        return self.project(
            out
        )


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads=4
    ):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            1,
            dim
        )

        self.attn = MDTA(
            dim,
            heads
        )

        self.norm2 = nn.GroupNorm(
            1,
            dim
        )

        self.ffn = nn.Sequential(

            nn.Conv2d(
                dim,
                dim * 2,
                1
            ),

            nn.GELU(),

            nn.Conv2d(
                dim * 2,
                dim,
                1
            ),

        )

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


class RestormerLite(nn.Module):

    def __init__(
        self,
        dim=48,
        blocks=6
    ):

        super().__init__()

        self.input = nn.Conv2d(
            3,
            dim,
            3,
            padding=1
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim,
                    heads=4
                )
                for _ in range(blocks)
            ]
        )

        self.output = nn.Conv2d(
            dim,
            3,
            3,
            padding=1
        )

    def forward(self, x):

        residual = x

        x = self.input(x)

        x = self.blocks(x)

        x = self.output(x)

        return torch.clamp(
            x + residual,
            0,
            1
        )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(name):

    if name == "unet":

        return UNet()

    if name == "nafnet":

        return NAFNet(
            width=32,
            blocks=8
        )

    if name == "restormer":

        return RestormerLite(
            dim=48,
            blocks=6
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0

    count = 0

    for degraded, gt, _ in loader:

        degraded = degraded.to(
            DEVICE,
            non_blocking=True
        )

        gt = gt.to(
            DEVICE,
            non_blocking=True
        )

        restored = model(
            degraded
        )

        loss = criterion(
            restored,
            gt
        )

        total_loss += (
            loss.item()
            * degraded.size(0)
        )

        count += degraded.size(0)

    return total_loss / count


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model_name,
    train_loader,
    val_loader
):

    print("\n")
    print("=" * 80)
    print(
        "TRAINING:",
        model_name.upper()
    )
    print("=" * 80)

    model = create_model(
        model_name
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_GRAD_SCALER
    )

    best_val = float("inf")

    patience_counter = 0

    history = []

    checkpoint = (
        CHECKPOINT_DIR /
        f"{model_name}_best.pth"
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0

        samples = 0

        start = time.time()

        for degraded, gt, _ in train_loader:

            degraded = degraded.to(
                DEVICE,
                non_blocking=True
            )

            gt = gt.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                dtype=AMP_DTYPE
            ):

                restored = model(
                    degraded
                )

                loss = criterion(
                    restored,
                    gt
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
                * degraded.size(0)
            )

            samples += degraded.size(0)

        train_loss = (
            running_loss /
            samples
        )

        val_loss = validate(
            model,
            val_loader
        )

        scheduler.step(
            val_loss
        )

        elapsed = (
            time.time()
            - start
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "learning_rate": current_lr,

            "time_sec": elapsed,

        })

        if val_loss < best_val:

            best_val = val_loss

            patience_counter = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                checkpoint
            )

            print(
                "  -> saved best model"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    pd.DataFrame(
        history
    ).to_csv(
        METRICS_DIR /
        f"{model_name}_training_history.csv",
        index=False
    )

    # Load best
    ckpt = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        ckpt["model"]
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    mse = np.mean(
        (
            prediction -
            target
        ) ** 2
    )

    if mse == 0:

        return 100.0

    return 10 * np.log10(
        1.0 / mse
    )


def calculate_ssim(
    prediction,
    target
):

    return structural_similarity(
        target,
        prediction,
        channel_axis=2,
        data_range=1.0
    )


# ============================================================
# OPTIONAL LPIPS
# ============================================================

try:

    import lpips

    LPIPS_AVAILABLE = True

    lpips_model = lpips.LPIPS(
        net="alex"
    ).to(DEVICE)

    lpips_model.eval()

    print(
        "LPIPS available."
    )

except Exception:

    LPIPS_AVAILABLE = False

    lpips_model = None

    print(
        "LPIPS unavailable."
    )


@torch.no_grad()
def calculate_lpips(
    prediction,
    target
):

    if not LPIPS_AVAILABLE:

        return np.nan

    p = torch.from_numpy(
        prediction
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    ).float()

    t = torch.from_numpy(
        target
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    ).float()

    p = p.to(DEVICE)

    t = t.to(DEVICE)

    # LPIPS expects [-1,1]
    p = p * 2 - 1

    t = t * 2 - 1

    value = lpips_model(
        p,
        t
    )

    return value.item()


# ============================================================
# TEST
# ============================================================

@torch.no_grad()
def evaluate_model(
    model_name,
    model,
    test_loader
):

    print("\n")
    print("=" * 80)

    print(
        "TESTING:",
        model_name.upper()
    )

    print("=" * 80)

    model.eval()

    results = []

    output_dir = (
        OUTPUT_DIR /
        model_name
    )

    for degraded, gt, filenames in test_loader:

        degraded = degraded.to(
            DEVICE
        )

        start = time.perf_counter()

        restored = model(
            degraded
        )

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        restored_np = (
            restored
            .cpu()
            .numpy()
            .transpose(
                0,
                2,
                3,
                1
            )
        )

        degraded_np = (
            degraded
            .cpu()
            .numpy()
            .transpose(
                0,
                2,
                3,
                1
            )
        )

        gt_np = (
            gt
            .numpy()
            .transpose(
                0,
                2,
                3,
                1
            )
        )

        for i, filename in enumerate(
            filenames
        ):

            prediction = np.clip(
                restored_np[i],
                0,
                1
            )

            target = np.clip(
                gt_np[i],
                0,
                1
            )

            degraded_image = np.clip(
                degraded_np[i],
                0,
                1
            )

            psnr = calculate_psnr(
                prediction,
                target
            )

            ssim = calculate_ssim(
                prediction,
                target
            )

            lpips_value = (
                calculate_lpips(
                    prediction,
                    target
                )
            )

            # Save restored image
            restored_image = Image.fromarray(
                (
                    prediction * 255
                ).astype(
                    np.uint8
                )
            )

            restored_image.save(
                output_dir /
                filename
            )

            results.append({

                "filename": filename,

                "PSNR": psnr,

                "SSIM": ssim,

                "LPIPS": lpips_value,

                "inference_ms":
                    elapsed /
                    len(filenames) *
                    1000,

            })

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        METRICS_DIR /
        f"{model_name}_test_metrics.csv",
        index=False
    )

    print(
        "\nAverage results:"
    )

    print(
        results_df[
            [
                "PSNR",
                "SSIM",
                "LPIPS",
                "inference_ms"
            ]
        ].mean()
    )

    return results_df


# ============================================================
# COMPARISON IMAGES
# ============================================================

def create_comparison(
    models,
    num_images=10
):

    print(
        "\nCreating comparison images..."
    )

    test_gt_dir = (
        GT_DIR /
        "test"
    )

    test_deg_dir = (
        DEGRADED_DIR /
        "test"
    )

    files = sorted(
        [
            p
            for p in test_gt_dir.iterdir()
            if p.suffix == ".png"
        ]
    )

    files = files[
        :num_images
    ]

    for gt_path in files:

        filename = gt_path.name

        gt = Image.open(
            gt_path
        ).convert("RGB")

        degraded = Image.open(
            test_deg_dir /
            filename
        ).convert("RGB")

        images = [
            ("GT", gt),
            ("Degraded", degraded),
        ]

        for model_name in models:

            path = (
                OUTPUT_DIR /
                model_name /
                filename
            )

            if path.exists():

                img = Image.open(
                    path
                ).convert("RGB")

                images.append(
                    (
                        model_name,
                        img
                    )
                )

        # Canvas
        width = IMAGE_SIZE

        label_height = 35

        canvas = Image.new(
            "RGB",
            (
                width * len(images),
                IMAGE_SIZE +
                label_height
            ),
            "white"
        )

        from PIL import ImageDraw

        draw = ImageDraw.Draw(
            canvas
        )

        for i, (label, image) in enumerate(
            images
        ):

            x = i * width

            canvas.paste(
                image,
                (
                    x,
                    label_height
                )
            )

            draw.text(
                (
                    x + 10,
                    8
                ),
                label,
                fill="black"
            )

        canvas.save(
            COMPARISON_DIR /
            f"comparison_{filename}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. CREATE DATASET
    # --------------------------------------------------------

    prepare_dataset()

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    train_dataset = RestorationDataset(

        DEGRADED_DIR / "train",

        GT_DIR / "train",

        augment=True

    )

    val_dataset = RestorationDataset(

        DEGRADED_DIR / "val",

        GT_DIR / "val",

        augment=False

    )

    test_dataset = RestorationDataset(

        DEGRADED_DIR / "test",

        GT_DIR / "test",

        augment=False

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=(
            NUM_WORKERS > 0
        ),

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True,

    )

    print("\nDataset sizes:")

    print(
        "Train:",
        len(train_dataset)
    )

    print(
        "Val:",
        len(val_dataset)
    )

    print(
        "Test:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # 3. TRAIN
    # --------------------------------------------------------

    trained_models = {}

    all_metrics = {}

    for model_name in MODELS_TO_TRAIN:

        model = train_model(

            model_name,

            train_loader,

            val_loader

        )

        trained_models[
            model_name
        ] = model

        # ----------------------------------------------------
        # 4. TEST
        # ----------------------------------------------------

        metrics = evaluate_model(

            model_name,

            model,

            test_loader

        )

        all_metrics[
            model_name
        ] = metrics

    # --------------------------------------------------------
    # 5. VISUAL COMPARISONS
    # --------------------------------------------------------

    create_comparison(

        MODELS_TO_TRAIN,

        num_images=20

    )

    # --------------------------------------------------------
    # 6. SUMMARY
    # --------------------------------------------------------

    summary = []

    for model_name, df in all_metrics.items():

        summary.append({

            "Model": model_name,

            "PSNR":
                df["PSNR"].mean(),

            "SSIM":
                df["SSIM"].mean(),

            "LPIPS":
                df["LPIPS"].mean(),

            "Inference_ms":
                df["inference_ms"].mean(),

        })

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        METRICS_DIR /
        "FINAL_MODEL_COMPARISON.csv",
        index=False
    )

    print("\n")
    print("=" * 80)

    print("FINAL RESULTS")

    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nComplete.")


if __name__ == "__main__":

    main()