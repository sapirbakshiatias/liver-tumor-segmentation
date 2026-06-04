"""
Extract deep features from liver CT volumes using a 3D U-Net autoencoder.

Approach:
  1. Train a 3D convolutional autoencoder (encoder + decoder) to RECONSTRUCT
     each liver volume — no labels used, purely unsupervised.
  2. After training, use only the ENCODER part to compress each volume
     into a fixed-size feature vector (bottleneck = 64 values).
  3. Save one feature vector per series -> unet_features.csv

Why this works:
  The encoder learns to capture the most important structural patterns
  in the liver (texture, shape, heterogeneity) in order to reconstruct it.
  Cancer and healthy livers look structurally different -> different encodings.

Architecture:
  Input:  (1, 64, 64, 32) — normalized liver volume
  Encoder: Conv3d 1->16->32->64, MaxPool at each step
  Bottleneck: GlobalAvgPool -> 64-d feature vector
  Decoder: ConvTranspose3d 64->32->16->1 (for training only)
  Loss: MSE reconstruction

Run: .venv\Scripts\python.exe data\extract_unet_features.py
"""
import sys, os, warnings
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CT_DIR   = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\All_Series_CT"
MASK_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\All_Series_Masks"
OUT_CSV  = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\unet_features.csv"

# Volume size fed into the network (smaller = faster on CPU)
VOL_SIZE = (64, 64, 32)
EPOCHS   = 60
LR       = 1e-3
BOTTLENECK = 64   # feature vector size per series

PATIENT_META = {
    "Patient_1_After":    {"group": "Patient_1",  "label": 1},
    "Patient_1_Before":   {"group": "Patient_1",  "label": 1},
    "Patient_2_After":    {"group": "Patient_2",  "label": 1},
    "Patient_2_Before":   {"group": "Patient_2",  "label": 1},
    "Patient_KB_Cancer":  {"group": "Patient_KB", "label": 1},
    "Patient_KB_Before":  {"group": "Patient_KB", "label": 1},
    "Patient_GA_Healthy": {"group": "Patient_GA", "label": 0},
    "Patient_VT_Healthy": {"group": "Patient_VT", "label": 0},
}

import re

def series_to_patient_key(stem):
    m = re.match(r'^(.+)_(s\d+)$', stem)
    return m.group(1) if m else None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_resize(ct_path, mask_path, vol_size):
    """Load a cropped liver CT, mask to liver region, normalize, resize."""
    ct   = nib.load(ct_path).get_fdata().astype(float)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)

    # Apply liver mask (zero out non-liver)
    ct[mask == 0] = -100

    # Clip HU and normalize to [0, 1]
    ct = np.clip(ct, -100, 400)
    ct = (ct - (-100)) / (400 - (-100))

    # Resize to fixed size using simple sampling
    from scipy.ndimage import zoom
    zoom_factors = (vol_size[0]/ct.shape[0],
                    vol_size[1]/ct.shape[1],
                    vol_size[2]/ct.shape[2])
    ct_resized = zoom(ct, zoom_factors, order=1)
    return ct_resized.astype(np.float32)


class LiverDataset(Dataset):
    def __init__(self, volumes):
        self.volumes = volumes  # list of (D, H, W) float32 arrays

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        # Add channel dim: (1, D, H, W)
        return torch.from_numpy(self.volumes[idx]).unsqueeze(0)


# ── U-Net Autoencoder ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1,  16)   # (1,64,64,32) -> (16,64,64,32)
        self.pool1  = nn.MaxPool3d(2)     # -> (16,32,32,16)
        self.block2 = ConvBlock(16, 32)   # -> (32,32,32,16)
        self.pool2  = nn.MaxPool3d(2)     # -> (32,16,16,8)
        self.block3 = ConvBlock(32, 64)   # -> (64,16,16,8)
        self.pool3  = nn.MaxPool3d(2)     # -> (64,8,8,4)
        self.gap    = nn.AdaptiveAvgPool3d(1)  # -> (64,1,1,1)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.gap(x)
        return x.view(x.size(0), -1)  # (B, 64)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Expand bottleneck back to spatial volume
        self.expand = nn.Linear(BOTTLENECK, 64 * 8 * 8 * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16,  1, 2, stride=2),
            nn.Sigmoid())

    def forward(self, z):
        x = self.expand(z).view(-1, 64, 8, 8, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all series
    print("\nLoading and resizing liver volumes...")
    series_list = []
    volumes     = []
    meta_rows   = []

    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.endswith(".nii.gz"):
            continue
        stem = fname.removeprefix("cropped_").removesuffix(".nii.gz")
        pkey = series_to_patient_key(stem)
        if pkey not in PATIENT_META:
            continue
        mask_fname = f"cropped_{stem}_mask.nii.gz"
        mask_path  = os.path.join(MASK_DIR, mask_fname)
        if not os.path.exists(mask_path):
            continue

        vol = load_and_resize(os.path.join(CT_DIR, fname), mask_path, VOL_SIZE)
        series_list.append(stem)
        volumes.append(vol)
        meta = PATIENT_META[pkey]
        meta_rows.append({"series": stem, "patient": pkey,
                          "group": meta["group"], "label": meta["label"]})
        print(f"  {stem:<40}  {vol.shape}")

    print(f"\n{len(volumes)} series loaded. Volume shape: {VOL_SIZE}")

    # Train autoencoder
    dataset    = LiverDataset(volumes)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model     = UNetAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    print(f"\nTraining U-Net autoencoder ({EPOCHS} epochs)...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg = total_loss / len(volumes)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg:.5f}")

    # Extract features
    print("\nExtracting bottleneck features...")
    model.eval()
    all_feats = []
    with torch.no_grad():
        for vol in volumes:
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            z = model.encoder(x)
            all_feats.append(z.cpu().numpy().flatten())

    all_feats = np.array(all_feats)  # (28, 64)

    # Save CSV
    feat_names = [f"unet_{i:03d}" for i in range(BOTTLENECK)]
    df = pd.DataFrame(meta_rows)
    for i, name in enumerate(feat_names):
        df[name] = all_feats[:, i]

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Shape: {df.shape}  ({len(df)} series x {BOTTLENECK} U-Net features)")
    print(f"\nLabel distribution:")
    for grp, sub in df.groupby("group"):
        lbl = "Cancer" if sub["label"].iloc[0]==1 else "Healthy"
        print(f"  {grp:<15} [{lbl}]  {len(sub)} series")
