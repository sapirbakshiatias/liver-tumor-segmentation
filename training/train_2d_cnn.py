"""
2D CNN pipeline on raw axial CT slices — patient-level LOOCV.

Strategy: ResNet18 as frozen feature extractor (run once, fast on CPU),
then train lightweight classifiers + attention heads on the 512-d features.

Models:
  1. SimpleCNN          — small 4-layer CNN, trained from scratch (112x112)
  2. ResNet18-Linear    — frozen ResNet18 backbone + Linear head
  3. ResNet18-SE        — frozen ResNet18 backbone + SE attention + Linear head
  4. ResNet18-FineTune  — ResNet18 last block unfrozen + Linear head
  5. Grad-CAM           — spatial attention maps for best ResNet model

LOOCV: all slices of one patient = test, rest = train.
       class-weighted BCELoss (12:1 imbalance).
       patient prediction = weighted avg of per-slice probs (weight=liver_px).

Run: .venv\Scripts\python.exe train_2d_cnn.py
"""

import sys, os; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")


import os, re, sys, warnings, time
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.models as tvm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CT_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_DIR  = os.path.join(BASE_DIR, "results"); os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE     = 112      # smaller = faster SimpleCNN
IMG_RESNET   = 224      # ResNet expects 224
MIN_LIVER_PX = 200
SLICE_STRIDE = 3
CNN_EPOCHS   = 20
FT_EPOCHS    = 15       # fine-tune epochs
BATCH        = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
PATIENTS = ["Patient_1", "Patient_2", "Patient_KB", "Patient_GA", "Patient_VT"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_slices():
    print("Loading NIfTI slices...", flush=True)
    ct_files = sorted(f for f in os.listdir(CT_DIR) if f.endswith(".nii.gz"))
    slices = []
    for ct_fname in ct_files:
        stem = ct_fname.removeprefix("cropped_").removesuffix(".nii.gz")
        m = re.match(r'^(.+)_(s\d+)$', stem)
        if not m: continue
        patient_key = m.group(1)
        if patient_key not in PATIENT_META: continue
        mask_path = os.path.join(MASK_DIR, f"cropped_{stem}_mask.nii.gz")
        if not os.path.exists(mask_path): continue
        ct_3d   = nib.load(os.path.join(CT_DIR, ct_fname)).get_fdata().astype(np.float32)
        mask_3d = nib.load(mask_path).get_fdata().astype(np.uint8)
        meta    = PATIENT_META[patient_key]
        for z in range(0, ct_3d.shape[2], SLICE_STRIDE):
            msk = mask_3d[:, :, z]
            lpx = int(msk.sum())
            if lpx < MIN_LIVER_PX: continue
            slices.append({"img": ct_3d[:, :, z], "mask": msk,
                           "label": meta["label"], "group": meta["group"],
                           "series": stem, "liver_px": lpx, "slice_idx": z})
    print(f"  {len(slices)} slices, {len(PATIENTS)} patients", flush=True)
    return slices


def to_tensor(sl, size):
    img = np.clip(sl, -100, 400)
    img = (img + 100) / 500.0
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    t = TF.resize(t, [size, size], antialias=True).repeat(3, 1, 1)
    mu  = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t - mu) / std


class SliceDS(Dataset):
    def __init__(self, slices, size):
        self.slices = slices; self.size = size
    def __len__(self): return len(self.slices)
    def __getitem__(self, i):
        s = self.slices[i]
        return to_tensor(s["img"], self.size), torch.tensor(s["label"], dtype=torch.float32), s["liver_px"]


# ── Models ────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(256*16, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,1),
        )
    def forward(self, x): return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns which of the 512 ResNet channels matter."""
    def __init__(self, c=512, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):   # x: (B, 512)
        return x * self.fc(x)


class ResNetHead(nn.Module):
    """Classifier head that optionally includes SE attention."""
    def __init__(self, use_se=False):
        super().__init__()
        self.se   = SEBlock() if use_se else nn.Identity()
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(512, 1)
    def forward(self, x):
        return self.fc(self.drop(self.se(x)))


# ── Feature extraction (ResNet18 backbone, run ONCE) ─────────────────────────

def extract_features(slices, size=IMG_RESNET):
    """Run frozen ResNet18 over all slices once, return (N, 512) feature matrix."""
    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(DEVICE)
    for p in backbone.parameters(): p.requires_grad_(False)

    ds     = SliceDS(slices, size)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    feats  = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            feats.append(backbone(imgs.to(DEVICE)).cpu())
    return torch.cat(feats, dim=0)   # (N, 512)


# ── Grad-CAM (for fine-tune model) ───────────────────────────────────────────

class GradCAM:
    def __init__(self, model, layer):
        self.act = self.grad = None
        layer.register_forward_hook(lambda m,i,o: setattr(self, "act", o.detach()))
        layer.register_full_backward_hook(lambda m,i,o: setattr(self, "grad", o[0].detach()))
    def compute(self, logit):
        logit.backward(retain_graph=True)
        w   = self.grad.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * self.act).sum(dim=1)).squeeze()
        cam = cam - cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        return cam.cpu().numpy()


# ── Training helpers ──────────────────────────────────────────────────────────

def bce_weighted(n_pos, n_neg):
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def train_cnn(model, slices, epochs, lr, img_size):
    ds     = SliceDS(slices, img_size)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    n_pos  = sum(s["label"] for s in slices)
    n_neg  = len(slices) - n_pos
    crit   = bce_weighted(n_pos, n_neg)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for _ in range(epochs):
        for imgs, lbls, _ in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()
        sched.step()
    return model


def train_head(head, feats_t, labels_t, epochs=40, lr=1e-3):
    n_pos  = int(labels_t.sum()); n_neg = len(labels_t) - n_pos
    crit   = bce_weighted(n_pos, n_neg)
    opt    = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    head.train()
    for _ in range(epochs):
        idx   = torch.randperm(len(feats_t))
        for b in range(0, len(feats_t), BATCH):
            bi = idx[b:b+BATCH]
            xb = feats_t[bi].to(DEVICE)
            yb = labels_t[bi].to(DEVICE).unsqueeze(1)
            opt.zero_grad(); crit(head(xb), yb).backward(); opt.step()
        sched.step()
    return head


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_slices_cnn(model, slices, img_size):
    ds     = SliceDS(slices, img_size)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    probs, lx = [], []
    model.eval()
    with torch.no_grad():
        for imgs, _, lpx in loader:
            probs.append(torch.sigmoid(model(imgs.to(DEVICE)).squeeze(1)).cpu().numpy())
            lx.append(lpx.numpy())
    return np.concatenate(probs), np.concatenate(lx).astype(float)


def predict_slices_head(head, feats, lx_arr):
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(DEVICE)).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
    return probs, lx_arr.astype(float)


def patient_prob(probs, lx):
    w = lx / lx.max()
    return float(np.average(probs, weights=w))


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(yt, yp, yprob=None):
    yt, yp = np.array(yt), np.array(yp)
    tp=int(((yt==1)&(yp==1)).sum()); tn=int(((yt==0)&(yp==0)).sum())
    fp=int(((yt==0)&(yp==1)).sum()); fn=int(((yt==1)&(yp==0)).sum())
    acc=  (tp+tn)/len(yt)
    sens= tp/(tp+fn)   if tp+fn>0 else float("nan")
    spec= tn/(tn+fp)   if tn+fp>0 else float("nan")
    prec= tp/(tp+fp)   if tp+fp>0 else float("nan")
    f1=   2*prec*sens/(prec+sens) if not any(np.isnan([prec,sens])) and prec+sens>0 else float("nan")
    auc=  float("nan")
    if yprob and len(np.unique(yt))==2:
        try: auc=roc_auc_score(yt, yprob)
        except: pass
    return acc, sens, spec, auc, f1


# ── LOOCV runners ─────────────────────────────────────────────────────────────

def loocv_cnn(model_name, all_slices, img_size, epochs, lr):
    ft, fp_l, fprob = [], [], []
    pp = {}
    for patient in PATIENTS:
        tr = [s for s in all_slices if s["group"] != patient]
        te = [s for s in all_slices if s["group"] == patient]
        tl = te[0]["label"]
        model = SimpleCNN().to(DEVICE)
        model = train_cnn(model, tr, epochs, lr, img_size)
        probs, lx = predict_slices_cnn(model, te, img_size)
        prob = patient_prob(probs, lx)
        pred = int(prob >= 0.5)
        ft.append(tl); fp_l.append(pred); fprob.append(prob)
        pp[patient] = (tl, pred, prob, len(te))
        ok = "OK" if tl==pred else "WRONG"
        print(f"    {patient:<15}  true={'Cancer' if tl else 'Healthy':<7}  prob={prob:.3f}  {ok}", flush=True)
    acc, sens, spec, auc, f1 = metrics(ft, fp_l, fprob)
    return acc, sens, spec, auc, f1, pp


def loocv_resnet(all_slices, all_feats, use_se, finetune=False):
    ft, fp_l, fprob = [], [], []
    pp = {}
    grp_arr = np.array([s["group"] for s in all_slices])
    lbl_arr = torch.tensor([s["label"] for s in all_slices], dtype=torch.float32)
    lx_arr  = np.array([s["liver_px"] for s in all_slices], dtype=float)

    for patient in PATIENTS:
        tr_mask = grp_arr != patient
        te_mask = grp_arr == patient
        tl      = int(lbl_arr[te_mask][0].item())

        if finetune:
            # Unfreeze layer4 of a fresh ResNet, fine-tune end-to-end
            backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            for p in list(backbone.children())[:7]:        # freeze up to layer3
                for w in p.parameters(): w.requires_grad_(False)
            backbone.fc = nn.Linear(512, 1)
            backbone = backbone.to(DEVICE)
            train_slices = [s for s in all_slices if s["group"] != patient]
            backbone = train_cnn(backbone, train_slices, FT_EPOCHS, 1e-4, IMG_RESNET)
            probs, lx = predict_slices_cnn(backbone, [s for s in all_slices if s["group"]==patient], IMG_RESNET)
        else:
            head = ResNetHead(use_se=use_se).to(DEVICE)
            head = train_head(head, all_feats[tr_mask], lbl_arr[tr_mask])
            probs, lx = predict_slices_head(head, all_feats[te_mask], lx_arr[te_mask])

        prob = patient_prob(probs, lx)
        pred = int(prob >= 0.5)
        ft.append(tl); fp_l.append(pred); fprob.append(prob)
        pp[patient] = (tl, pred, prob, int(te_mask.sum()))
        ok = "OK" if tl==pred else "WRONG"
        print(f"    {patient:<15}  true={'Cancer' if tl else 'Healthy':<7}  prob={prob:.3f}  {ok}", flush=True)

    acc, sens, spec, auc, f1 = metrics(ft, fp_l, fprob)
    return acc, sens, spec, auc, f1, pp


# ── Grad-CAM visualisation ────────────────────────────────────────────────────

def save_gradcam(all_slices):
    print("\nGenerating Grad-CAM...", flush=True)
    # Train on everyone except VT+GA, show on representative slices
    train_sl = [s for s in all_slices if s["group"] not in ("Patient_VT","Patient_GA")]
    show_sl  = []
    for grp in ("Patient_KB","Patient_GA","Patient_VT"):
        cands = [s for s in all_slices if s["group"]==grp]
        # Pick 3 slices spread through the volume
        step = max(1, len(cands)//3)
        show_sl.extend(cands[::step][:3])

    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    for p in list(backbone.children())[:7]:
        for w in p.parameters(): w.requires_grad_(False)
    backbone.fc = nn.Linear(512, 1)
    backbone = backbone.to(DEVICE)
    backbone = train_cnn(backbone, train_sl, FT_EPOCHS, 1e-4, IMG_RESNET)
    backbone.eval()

    gcam = GradCAM(backbone, backbone.layer4[-1].conv2)

    n = len(show_sl)
    fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
    color_map = {"Patient_KB":"#C62828","Patient_GA":"#1B5E20","Patient_VT":"#1E88E5"}

    for i, s in enumerate(show_sl):
        img_t = to_tensor(s["img"], IMG_RESNET).unsqueeze(0).to(DEVICE)
        img_t.requires_grad_(True)
        backbone.zero_grad()
        logit = backbone(img_t)
        cam   = gcam.compute(logit)

        sl_disp = np.clip(s["img"], -100, 400)
        sl_disp = (sl_disp - sl_disp.min()) / (sl_disp.max() - sl_disp.min() + 1e-10)

        # Upsample CAM to image size
        cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_up = nn.functional.interpolate(
            cam_t, size=sl_disp.shape, mode="bilinear", align_corners=False
        ).squeeze().numpy()

        axes[0,i].imshow(sl_disp.T, cmap="gray", origin="lower")
        axes[0,i].axis("off")
        axes[0,i].set_title(f"{s['group']}\n[{'Cancer' if s['label'] else 'Healthy'}]",
                            fontsize=7, color=color_map.get(s["group"],"black"), fontweight="bold")

        axes[1,i].imshow(sl_disp.T, cmap="gray", origin="lower")
        axes[1,i].imshow(cam_up.T, cmap="jet", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[1,i].axis("off")
        with torch.no_grad():
            prob = torch.sigmoid(backbone(img_t.detach())).item()
        axes[1,i].set_title(f"P(Cancer)={prob:.2f}", fontsize=8)

    fig.suptitle("Grad-CAM — ResNet18 fine-tuned\nTop: CT slice  |  Bottom: where the model looks (red=high attention)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "gradcam_resnet18.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}", flush=True)
    t0 = time.time()

    all_slices = load_all_slices()

    # Pre-extract ResNet features ONCE (all slices)
    print(f"\nExtracting ResNet18 features (frozen backbone, run once)...", flush=True)
    all_feats = extract_features(all_slices, size=IMG_RESNET)
    print(f"  Features: {all_feats.shape}  ({time.time()-t0:.0f}s so far)", flush=True)

    results = []

    experiments = [
        ("SimpleCNN",         "cnn"),
        ("ResNet18-Linear",   "resnet"),
        ("ResNet18-SE-Attn",  "resnet_se"),
        ("ResNet18-FineTune", "resnet_ft"),
    ]

    for model_name, mode in experiments:
        print(f"\n{'='*65}", flush=True)
        print(f"  {model_name}", flush=True)
        print(f"{'='*65}", flush=True)

        t1 = time.time()
        if mode == "cnn":
            acc,sens,spec,auc,f1,pp = loocv_cnn(
                model_name, all_slices, IMG_SIZE, CNN_EPOCHS, 5e-4)
        elif mode == "resnet":
            acc,sens,spec,auc,f1,pp = loocv_resnet(all_slices, all_feats, use_se=False)
        elif mode == "resnet_se":
            acc,sens,spec,auc,f1,pp = loocv_resnet(all_slices, all_feats, use_se=True)
        else:
            acc,sens,spec,auc,f1,pp = loocv_resnet(all_slices, all_feats, use_se=False, finetune=True)

        auc_s = f"{auc:.3f}" if auc==auc else "N/A"
        f1_s  = f"{f1:.3f}"  if f1==f1   else "N/A"
        wrong = [f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})"
                 for p in PATIENTS if pp[p][0]!=pp[p][1]]
        print(f"\n  => Acc={acc:.0%}  Sens={sens:.0%}  Spec={spec:.0%}  "
              f"AUC={auc_s}  F1={f1_s}  ({time.time()-t1:.0f}s)", flush=True)
        results.append((model_name, acc, sens, spec, auc, f1, pp, wrong))

    # Summary
    print(f"\n{'='*85}", flush=True)
    print("SUMMARY — 2D CNN models vs 3D classical", flush=True)
    print(f"{'='*85}", flush=True)
    print(f"  {'Model':<22} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>7} {'F1':>6}  Errors", flush=True)
    print(f"  {'-'*80}", flush=True)
    for name, acc, sens, spec, auc, f1, pp, wrong in results:
        auc_s = f"{auc:.3f}" if auc==auc else "  N/A"
        f1_s  = f"{f1:.3f}"  if f1==f1   else "  N/A"
        print(f"  {name:<22} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {auc_s:>7} {f1_s:>6}  "
              f"{', '.join(wrong) if wrong else 'none'}", flush=True)
    print(f"  {'3D VaRFS+KNN (ref)':<22}  100%  100%  100%  1.000  1.000  none", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)

    # Grad-CAM
    save_gradcam(all_slices)
    print("\nDone.", flush=True)
