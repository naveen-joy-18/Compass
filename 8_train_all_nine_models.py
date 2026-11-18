#!/usr/bin/env python
"""
CRLP: Nine-Model Self-Supervised Representation Learning Suite with
Comprehensive Classification, Reconstruction, and Diagnostic Metrics.

Models:
  1. TabNet-style attentive feature selection autoencoder
  2. SAINT-style row/column transformer autoencoder
  3. NODE-style differentiable oblivious-tree autoencoder
  4. GrowNet-style sequential residual boosting autoencoder
  5. DANet-style multi-branch attentive network autoencoder
  6. ModernNCA-style contrastive metric encoder with reconstruction head
  7. Variational Autoencoder (VAE)
  8. Deep Belief Network (stacked Gaussian-Bernoulli RBMs + fine-tuning)
  9. Deep Boltzmann Machine (two-hidden-layer energy model with mean-field inference)

Evaluated Metrics:
  Classification Metrics (via 5-Fold Stratified Cross-Validated Linear Probing):
    - AUROC, AUPRC
    - Accuracy, Precision, Recall, Specificity
    - F1-score, MCC (Matthews Correlation Coefficient)
    - Brier score
    - Confusion matrices (table, CSV, and multi-panel heatmap figure)
    - Calibration curves (table, CSV, and multi-line reliability diagrams)
    - ROC and Precision-Recall curves

  Representation and Reconstruction Metrics:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Validation Reconstruction Loss
    - Silhouette score
    - Davies–Bouldin index
    - Calinski–Harabasz index

  Training Diagnostics:
    - Epoch-wise training loss (epochs 1 to 100)
    - Epoch-wise validation loss on dedicated validation split
    - Training time per model (seconds)
    - GPU memory usage (peak MB, or CPU RAM usage if CUDA unavailable)

Usage:
  python 8_train_all_nine_models.py
  python 8_train_all_nine_models.py --epochs 100 --top-genes 256 --latent-dim 16
"""

from pathlib import Path
import argparse, json, random, time, copy, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.calibration import calibration_curve

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "perturbation_analysis" / "GSE90546_guide_pseudobulk_profiles.csv.gz"
OUT = ROOT / "training_all_models"
FIG = OUT / "figures"
MODELS = OUT / "models"


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_memory_info(device):
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {"device": "cuda", "peak_gpu_mb": float(round(gpu_mb, 2)), "summary": f"{gpu_mb:.2f} MB GPU"}
    elif HAS_PSUTIL:
        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        return {"device": "cpu", "peak_gpu_mb": 0.0, "process_ram_mb": float(round(rss_mb, 2)), "summary": f"N/A (CPU RAM: {rss_mb:.1f} MB)"}
    else:
        return {"device": "cpu", "peak_gpu_mb": 0.0, "summary": "N/A (CPU)"}


# ============================================================
# ARCHITECTURES
# ============================================================

class MLP(nn.Module):
    def __init__(self, d, h, out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, out)
        )
    def forward(self, x):
        return self.net(x)


# 1. TabNet-style attentive feature selection autoencoder
class TabNetAE(nn.Module):
    def __init__(self, d, z=16, h=128, steps=3):
        super().__init__()
        self.steps = steps
        self.attn = nn.ModuleList([nn.Linear(d, d) for _ in range(steps)])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(),
                nn.Linear(h, h), nn.GELU()
            ) for _ in range(steps)
        ])
        self.to_z = nn.Linear(h * steps, z)
        self.dec = MLP(z, h, d)

    def forward(self, x):
        prior = torch.ones_like(x)
        reps = []
        for a, b in zip(self.attn, self.blocks):
            mask = torch.softmax(a(x) * prior, dim=1)
            prior = prior * (1.0 - mask)
            reps.append(b(x * mask))
        z = self.to_z(torch.cat(reps, dim=1))
        return self.dec(z), z


# 2. SAINT-style feature-token transformer autoencoder
class SAINTAE(nn.Module):
    def __init__(self, d, z=16, dim=32, heads=4, layers=2):
        super().__init__()
        self.d = d
        self.val = nn.Linear(1, dim)
        self.pos = nn.Parameter(torch.randn(1, d, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 2,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
        self.pool = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.to_z = nn.Linear(dim, z)
        self.dec = MLP(z, 128, d)

    def forward(self, x):
        t = self.val(x.unsqueeze(-1)) + self.pos
        h = self.tr(t).mean(dim=1)
        z = self.to_z(self.pool(h))
        return self.dec(z), z


# 3. NODE-style differentiable oblivious-tree autoencoder
class ObliviousTreeLayer(nn.Module):
    def __init__(self, d, trees=16, depth=4):
        super().__init__()
        self.trees = trees
        self.depth = depth
        self.feature_logits = nn.Parameter(torch.randn(trees, depth, d) * 0.02)
        self.threshold = nn.Parameter(torch.zeros(trees, depth))
        self.leaves = nn.Parameter(torch.randn(trees, 2 ** depth, 16) * 0.05)

    def forward(self, x):
        probs = []
        for level in range(self.depth):
            w = torch.softmax(self.feature_logits[:, level, :], dim=-1)
            val = x @ w.T
            p = torch.sigmoid((val - self.threshold[:, level]) * 4.0)
            probs.append(p)
        leaf_prob = torch.ones(x.size(0), self.trees, 1, device=x.device)
        for p in probs:
            p = p.unsqueeze(-1)
            leaf_prob = torch.cat([leaf_prob * (1.0 - p), leaf_prob * p], dim=-1).reshape(x.size(0), self.trees, -1)
        out = torch.einsum("btl,tld->btd", leaf_prob, self.leaves)
        return out.mean(dim=1)


class NODEAE(nn.Module):
    def __init__(self, d, z=16):
        super().__init__()
        self.tree = ObliviousTreeLayer(d)
        self.proj = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, z))
        self.dec = MLP(z, 128, d)

    def forward(self, x):
        z = self.proj(self.tree(x))
        return self.dec(z), z


# 4. GrowNet-style stage-wise residual boosting autoencoder
class GrowNetAE(nn.Module):
    def __init__(self, d, z=16, stages=5, h=128):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(1, d))
        self.learners = nn.ModuleList([MLP(d, h, d, 0.05) for _ in range(stages)])
        self.encoder = MLP(d, h, z)

    def forward(self, x):
        pred = self.base.expand_as(x)
        for learner in self.learners:
            pred = pred + 0.2 * learner(x)
        z = self.encoder(pred)
        return pred, z


# 5. DANet-style parallel attentive branches autoencoder
class DANetAE(nn.Module):
    def __init__(self, d, z=16, branches=4, h=128):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
            for _ in range(branches)
        ])
        self.gates = nn.Linear(d, branches)
        self.to_z = nn.Linear(h * branches, z)
        self.dec = MLP(z, h, d)

    def forward(self, x):
        weights = torch.softmax(self.gates(x), dim=1)
        hs = [b(x) * weights[:, i:i+1] for i, b in enumerate(self.branches)]
        z = self.to_z(torch.cat(hs, dim=1))
        return self.dec(z), z


# 6. ModernNCA-style contrastive metric encoder with attached reconstruction head
class ModernNCAAE(nn.Module):
    def __init__(self, d, z=32, h=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, z)
        )
        self.dec = MLP(z, h, d)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=1)
        recon = self.dec(z)
        return recon, z


# 7. Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, d, z=16, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h // 2), nn.GELU())
        self.mu = nn.Linear(h // 2, z)
        self.lv = nn.Linear(h // 2, z)
        self.dec = MLP(z, h, d)

    def forward(self, x):
        q = self.enc(x)
        mu = self.mu(q)
        lv = self.lv(q).clamp(-8, 8)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        else:
            z = mu
        return self.dec(z), mu, lv


# 8. Deep Belief Network (Gaussian-Bernoulli Stacked RBMs)
class RBM(nn.Module):
    def __init__(self, nv, nh):
        super().__init__()
        self.W = nn.Parameter(torch.randn(nv, nh) * 0.01)
        self.vb = nn.Parameter(torch.zeros(nv))
        self.hb = nn.Parameter(torch.zeros(nh))

    def sample_h(self, v):
        p = torch.sigmoid(v @ self.W + self.hb)
        return p, torch.bernoulli(p)

    def sample_v(self, h):
        # Continuous linear visible units for standardized gene expression
        p = h @ self.W.T + self.vb
        return p, p

    def cd_loss(self, v, k=1):
        ph, _ = self.sample_h(v)
        h = torch.bernoulli(ph)
        for _ in range(k):
            pv, vn = self.sample_v(h)
            phn, hn = self.sample_h(vn)
            h = hn
        pos = (v @ self.W * ph).sum(dim=1) + v @ self.vb + ph @ self.hb
        neg = (vn @ self.W * phn).sum(dim=1) + vn @ self.vb + phn @ self.hb
        return -(pos - neg).mean()


class DBN(nn.Module):
    def __init__(self, d, z=16, h=128):
        super().__init__()
        self.rbm1 = RBM(d, h)
        self.rbm2 = RBM(h, z)

    def forward(self, x):
        h1 = torch.sigmoid(x @ self.rbm1.W + self.rbm1.hb)
        z = torch.sigmoid(h1 @ self.rbm2.W + self.rbm2.hb)
        h1r = torch.sigmoid(z @ self.rbm2.W.T + self.rbm2.vb)
        xr = h1r @ self.rbm1.W.T + self.rbm1.vb
        return xr, z


# 9. Deep Boltzmann Machine (Two-hidden-layer energy model with mean-field inference)
class DBM(nn.Module):
    def __init__(self, d, z=16, h=96):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d, h) * 0.01)
        self.W2 = nn.Parameter(torch.randn(h, z) * 0.01)
        self.vb = nn.Parameter(torch.zeros(d))
        self.hb = nn.Parameter(torch.zeros(h))
        self.zb = nn.Parameter(torch.zeros(z))

    def infer(self, x, steps=3):
        h = torch.sigmoid(x @ self.W1 + self.hb)
        z = torch.sigmoid(h @ self.W2 + self.zb)
        for _ in range(steps):
            h = torch.sigmoid(x @ self.W1 + z @ self.W2.T + self.hb)
            z = torch.sigmoid(h @ self.W2 + self.zb)
        return h, z

    def forward(self, x):
        h, z = self.infer(x)
        xr = h @ self.W1.T + self.vb
        return xr, z


def nt_xent(z1, z2, temp=0.2):
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temp
    n = z1.size(0)
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ============================================================
# TRAINING ROUTINES WITH TRAIN & VALIDATION LOSS TRACKING
# ============================================================

def train_reconstruction_model(name, model, X_train, X_val, device, epochs, batch_size, lr, beta=0.01):
    model.to(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=min(batch_size, len(X_train)),
        shuffle=True, drop_last=False
    )
    x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    hist_train = []
    hist_val = []

    start_time = time.perf_counter()

    for ep in range(epochs):
        model.train()
        total_train = 0.0
        n_train = 0

        for (x,) in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)

            # Denoising reconstruction: mask 10% of features
            mask = (torch.rand_like(x) < 0.10)
            xin = x.clone()
            xin[mask] = 0.0
            out = model(xin)

            if name == "VAE":
                recon, mu, lv = out
                rec = F.mse_loss(recon, x)
                kl = -0.5 * torch.mean(1.0 + lv - mu.pow(2) - lv.exp())
                loss = rec + beta * kl
            else:
                recon, z = out
                loss = F.mse_loss(recon, x)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train += loss.item() * len(x)
            n_train += len(x)

        ep_train_loss = total_train / n_train
        hist_train.append(ep_train_loss)

        # Validation loss evaluation
        model.eval()
        with torch.no_grad():
            out_val = model(x_val_t)
            if name == "VAE":
                recon_val, mu_val, lv_val = out_val
                rec_val = F.mse_loss(recon_val, x_val_t)
                kl_val = -0.5 * torch.mean(1.0 + lv_val - mu_val.pow(2) - lv_val.exp())
                ep_val_loss = (rec_val + beta * kl_val).item()
            else:
                recon_val, _ = out_val
                ep_val_loss = F.mse_loss(recon_val, x_val_t).item()
            hist_val.append(ep_val_loss)

        if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == epochs:
            print(f"  [{name}] Epoch {ep+1:03d}/{epochs} | Train Loss: {ep_train_loss:.6f} | Val Loss: {ep_val_loss:.6f}")

    train_time = round(time.perf_counter() - start_time, 2)
    mem_info = get_memory_info(device)

    # Full-dataset inference for final representation and reconstruction
    model.eval()
    with torch.no_grad():
        x_all = torch.tensor(np.concatenate([X_train, X_val], axis=0), dtype=torch.float32, device=device)
        # Note: caller will evaluate on full X directly
    return hist_train, hist_val, train_time, mem_info


def train_modernnca_model(model, X_train, X_val, device, epochs, batch_size, lr):
    model.to(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=min(batch_size, len(X_train)),
        shuffle=True, drop_last=False
    )
    x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    hist_train = []
    hist_val = []

    start_time = time.perf_counter()

    for ep in range(epochs):
        model.train()
        total_train = 0.0
        n_train = 0

        for (x,) in train_loader:
            x = x.to(device)
            # Two stochastic views
            x1 = x + 0.05 * torch.randn_like(x)
            x2 = x + 0.05 * torch.randn_like(x)
            opt.zero_grad(set_to_none=True)

            recon1, z1 = model(x1)
            recon2, z2 = model(x2)
            contrastive_loss = nt_xent(z1, z2)
            recon_loss = 0.5 * (F.mse_loss(recon1, x) + F.mse_loss(recon2, x))
            loss = contrastive_loss + recon_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train += loss.item() * len(x)
            n_train += len(x)

        ep_train_loss = total_train / n_train
        hist_train.append(ep_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            x1_v = x_val_t + 0.05 * torch.randn_like(x_val_t)
            x2_v = x_val_t + 0.05 * torch.randn_like(x_val_t)
            recon_v, z1_v = model(x1_v)
            _, z2_v = model(x2_v)
            v_contrast = nt_xent(z1_v, z2_v)
            v_recon = F.mse_loss(recon_v, x_val_t)
            ep_val_loss = (v_contrast + v_recon).item()
            hist_val.append(ep_val_loss)

        if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == epochs:
            print(f"  [ModernNCA] Epoch {ep+1:03d}/{epochs} | Train Loss: {ep_train_loss:.6f} | Val Loss: {ep_val_loss:.6f}")

    train_time = round(time.perf_counter() - start_time, 2)
    mem_info = get_memory_info(device)
    return hist_train, hist_val, train_time, mem_info


def train_dbn_model(model, X_train, X_val, device, epochs, batch_size, lr):
    model.to(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()

    # Pretrain RBM1 with CD-1
    print("  [DBN] Pretraining RBM layer 1 (Contrastive Divergence)...")
    opt_rbm1 = torch.optim.Adam(model.rbm1.parameters(), lr=1e-3)
    loader_r1 = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=min(batch_size, len(X_train)), shuffle=True)
    for _ in range(epochs // 2):
        for (x,) in loader_r1:
            x = x.to(device)
            opt_rbm1.zero_grad()
            loss1 = model.rbm1.cd_loss(x, k=1)
            loss1.backward()
            opt_rbm1.step()

    # Pretrain RBM2 on hidden activations
    print("  [DBN] Pretraining RBM layer 2 (Contrastive Divergence)...")
    with torch.no_grad():
        h1_train = torch.sigmoid(torch.tensor(X_train, dtype=torch.float32, device=device) @ model.rbm1.W + model.rbm1.hb)
    opt_rbm2 = torch.optim.Adam(model.rbm2.parameters(), lr=1e-3)
    loader_r2 = DataLoader(TensorDataset(h1_train), batch_size=min(batch_size, len(h1_train)), shuffle=True)
    for _ in range(epochs // 2):
        for (h,) in loader_r2:
            opt_rbm2.zero_grad()
            loss2 = model.rbm2.cd_loss(h, k=1)
            loss2.backward()
            opt_rbm2.step()

    # End-to-end fine-tuning
    print("  [DBN] End-to-end fine-tuning...")
    opt_dbn = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=min(batch_size, len(X_train)), shuffle=True)
    x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    hist_train = []
    hist_val = []

    for ep in range(epochs):
        model.train()
        total_train = 0.0
        n_train = 0
        for (x,) in train_loader:
            x = x.to(device)
            opt_dbn.zero_grad(set_to_none=True)
            xr, z = model(x)
            loss = F.mse_loss(xr, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_dbn.step()
            total_train += loss.item() * len(x)
            n_train += len(x)

        ep_train_loss = total_train / n_train
        hist_train.append(ep_train_loss)

        model.eval()
        with torch.no_grad():
            xr_v, _ = model(x_val_t)
            ep_val_loss = F.mse_loss(xr_v, x_val_t).item()
            hist_val.append(ep_val_loss)

        if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == epochs:
            print(f"  [DBN] Epoch {ep+1:03d}/{epochs} | Train Loss: {ep_train_loss:.6f} | Val Loss: {ep_val_loss:.6f}")

    train_time = round(time.perf_counter() - start_time, 2)
    mem_info = get_memory_info(device)
    return hist_train, hist_val, train_time, mem_info


def get_model_outputs(name, model, X, device):
    model.eval()
    with torch.no_grad():
        xx = torch.tensor(X, dtype=torch.float32, device=device)
        if name == "VAE":
            recon, mu, lv = model(xx)
            z = mu
        elif name == "ModernNCA":
            recon, z = model(xx)
        elif name == "DBN":
            recon, z = model(xx)
        elif name == "DBM":
            recon, z = model(xx)
        else:
            recon, z = model(xx)
        emb = z.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()
    return emb, recon_np


# ============================================================
# EVALUATION: CLASSIFICATION & CALIBRATION METRICS
# ============================================================

def evaluate_classification_probe(Z, y, task_name="Task", seed=42):
    """
    Performs 5-Fold Stratified Cross-Validation on the frozen representations Z
    using Logistic Regression to evaluate linear separability and representation quality.
    Computes all standard classification, reliability, and calibration metrics.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_probs = np.zeros(len(y), dtype=np.float64)
    y_preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in skf.split(Z, y):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(Z[train_idx], y[train_idx])
        probs = clf.predict_proba(Z[test_idx])[:, 1]
        y_probs[test_idx] = probs
        y_preds[test_idx] = (probs >= 0.5).astype(int)

    # Classification Metrics
    auroc = float(roc_auc_score(y, y_probs))
    auprc = float(average_precision_score(y, y_probs))
    acc = float(accuracy_score(y, y_preds))
    prec = float(precision_score(y, y_preds, zero_division=0))
    rec = float(recall_score(y, y_preds, zero_division=0))
    f1 = float(f1_score(y, y_preds, zero_division=0))
    mcc = float(matthews_corrcoef(y, y_preds))
    brier = float(brier_score_loss(y, y_probs))

    cm = confusion_matrix(y, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Calibration Curve (5 bins)
    prob_true, prob_pred = calibration_curve(y, y_probs, n_bins=5, strategy="uniform")

    # ROC and PR curve points
    fpr, tpr, _ = roc_curve(y, y_probs)
    pr_prec, pr_rec, _ = precision_recall_curve(y, y_probs)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
        "mcc": mcc,
        "brier_score": brier,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        },
        "cm_raw": cm.tolist(),
        "confusion_matrix_raw": cm.tolist(),
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        },
        "roc": {
            "fpr": fpr.tolist(), "tpr": tpr.tolist()
        },
        "pr": {
            "precision": pr_prec.tolist(), "recall": pr_rec.tolist()
        },
        "y_true": y.tolist(),
        "y_probs": y_probs.tolist(),
        "y_preds": y_preds.tolist()
    }
    return metrics


# ============================================================
# EVALUATION: RECONSTRUCTION & REPRESENTATION METRICS
# ============================================================

def evaluate_reconstruction_and_clustering(X, X_recon, Z, seed=42):
    # Reconstruction metrics
    mse = float(mean_squared_error(X, X_recon))
    mae = float(mean_absolute_error(X, X_recon))
    rmse = float(np.sqrt(mse))

    # Clustering metrics
    k = min(4, max(2, len(Z) // 20))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(Z)

    if len(np.unique(labels)) > 1:
        sil = float(silhouette_score(Z, labels))
        db = float(davies_bouldin_score(Z, labels))
        ch = float(calinski_harabasz_score(Z, labels))
    else:
        sil, db, ch = float("nan"), float("nan"), float("nan")

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "k_clusters": k,
        "silhouette_score": sil,
        "davies_bouldin_index": db,
        "calinski_harabasz_index": ch,
        "cluster_labels": labels
    }


# ============================================================
# PLOTTING FUNCTIONS (SVG & PNG)
# ============================================================

def save_dual_figure(fig, path_base):
    fig.tight_layout()
    fig.savefig(f"{path_base}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{path_base}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_confusion_matrices(cls_results, fig_dir):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(cls_results.items()):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix_raw"])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=9)
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f} | F1: {res['f1_score']:.3f} | MCC: {res['mcc']:.3f}", fontsize=10)

    save_dual_figure(fig, fig_dir / "all_models_confusion_matrices")


def plot_all_calibration_curves(cls_results, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

    colors = plt.cm.tab10(np.linspace(0, 1, len(cls_results)))
    for (name, res), color in zip(cls_results.items(), colors):
        cal = res["calibration"]
        ax.plot(cal["prob_pred"], cal["prob_true"], marker="o", lw=1.8, color=color,
                label=f"{name} (Brier: {res['brier_score']:.4f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Empirical)", fontsize=11)
    ax.set_title("Calibration Curves (Reliability Diagrams) Across All Nine Models", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    save_dual_figure(fig, fig_dir / "all_models_calibration_curves")


def plot_all_roc_and_pr_curves(cls_results, fig_dir):
    # ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Level (AUROC = 0.50)")
    colors = plt.cm.tab10(np.linspace(0, 1, len(cls_results)))
    for (name, res), color in zip(cls_results.items(), colors):
        roc = res["roc"]
        ax.plot(roc["fpr"], roc["tpr"], lw=1.8, color=color,
                label=f"{name} (AUROC = {res['auroc']:.3f})")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Across All Nine Models", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    save_dual_figure(fig, fig_dir / "all_models_roc_curves")

    # Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, res), color in zip(cls_results.items(), colors):
        pr = res["pr"]
        ax.plot(pr["recall"], pr["precision"], lw=1.8, color=color,
                label=f"{name} (AUPRC = {res['auprc']:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curves Across All Nine Models", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    save_dual_figure(fig, fig_dir / "all_models_pr_curves")


def plot_training_validation_losses(train_histories, val_histories, fig_dir):
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(train_histories.keys()):
        ax = axes[idx]
        tr = train_histories[name]
        va = val_histories[name]
        epochs = np.arange(1, len(tr) + 1)
        ax.plot(epochs, tr, label="Train Loss", color="#1f77b4", lw=1.8)
        ax.plot(epochs, va, label="Validation Loss", color="#ff7f0e", lw=1.8, linestyle="--")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        if idx >= 6:
            ax.set_xlabel("Epoch", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Objective Loss", fontsize=10)

    save_dual_figure(fig, fig_dir / "all_models_training_validation_losses")


def plot_metric_barcharts(summary_df, fig_dir):
    # 1. Classification Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.16
    metrics_to_plot = ["auroc", "auprc", "accuracy", "f1_score", "mcc"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (m, col) in enumerate(zip(metrics_to_plot, colors)):
        ax.bar(x + (i - 2) * width, summary_df[m], width, label=m.upper(), color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Classification & Discriminative Power (5-Fold CV Linear Probe)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    save_dual_figure(fig, fig_dir / "all_models_classification_metrics")

    # 2. Reconstruction Metrics Comparison (MSE, MAE, RMSE)
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_recon = ["mse", "mae", "rmse"]
    colors_recon = ["#e377c2", "#7f7f7f", "#bcbd22"]
    width = 0.22
    for i, (m, col) in enumerate(zip(metrics_recon, colors_recon)):
        ax.bar(x + (i - 1) * width, summary_df[m], width, label=m.upper(), color=col, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Reconstruction Error", fontsize=11)
    ax.set_title("Reconstruction Errors (MSE, MAE, RMSE) Across Nine Models", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save_dual_figure(fig, fig_dir / "all_models_reconstruction_metrics")

    # 3. Clustering Metrics Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    ax1.bar(summary_df["model"], summary_df["silhouette_score"], color="#17becf", alpha=0.85)
    ax1.set_title("Silhouette Score (Higher is better)", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", rotation=40)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(summary_df["model"], summary_df["davies_bouldin_index"], color="#8c564b", alpha=0.85)
    ax2.set_title("Davies–Bouldin Index (Lower is better)", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=40)
    ax2.grid(True, axis="y", alpha=0.25)

    ax3.bar(summary_df["model"], summary_df["calinski_harabasz_index"], color="#e377c2", alpha=0.85)
    ax3.set_title("Calinski–Harabasz Index (Higher is better)", fontsize=10, fontweight="bold")
    ax3.tick_params(axis="x", rotation=40)
    ax3.grid(True, axis="y", alpha=0.25)

    save_dual_figure(fig, fig_dir / "all_models_clustering_metrics")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CRLP Nine-Model Complete Representation Learning Suite")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--top-genes", type=int, default=256, help="Top variable genes to use (default: 256)")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent embedding dimension (default: 16)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation split fraction (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    seed_all(args.seed)
    OUT.mkdir(exist_ok=True, parents=True)
    FIG.mkdir(exist_ok=True, parents=True)
    MODELS.mkdir(exist_ok=True, parents=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing required input file: {INPUT}")

    print("=" * 75)
    print("CRLP: NINE-MODEL COMPREHENSIVE TRAINING & BENCHMARKING SUITE")
    print(f"Target Epochs : {args.epochs}")
    print(f"Top Genes     : {args.top_genes}")
    print(f"Latent Dim    : {args.latent_dim}")
    print(f"Batch Size    : {args.batch_size}")
    print("=" * 75)

    # 1. Load Data
    print("\n[1/6] Loading guide-level pseudobulk data...")
    df = pd.read_csv(INPUT, compression="gzip", low_memory=False)
    needed = {"gene_symbol", "sample_id", "guide_identity", "log2_CPM"}
    if needed - set(df.columns):
        raise ValueError(f"Missing required columns in input: {sorted(needed - set(df.columns))}")

    df = df.dropna(subset=["gene_symbol", "log2_CPM"]).copy()
    df["profile_id"] = df["sample_id"].astype(str) + "::" + df["guide_identity"].astype(str)
    mat = df.pivot_table(index="profile_id", columns="gene_symbol", values="log2_CPM", aggfunc="mean")
    mat = mat.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0)

    meta_cols = [c for c in ["sample_id", "guide_identity", "target_symbols", "guide_type",
                             "is_negative_control", "mapping_verified", "n_cells"] if c in df.columns]
    meta = df[["profile_id"] + meta_cols].drop_duplicates("profile_id").set_index("profile_id").reindex(mat.index).reset_index()

    # Center within each pooled sample to eliminate baseline sample-to-sample shift
    sm = meta.set_index("profile_id").loc[mat.index, "sample_id"]
    centered = mat.copy()
    for sample in sm.unique():
        ids = sm[sm == sample].index
        centered.loc[ids] = centered.loc[ids] - centered.loc[ids].mean(axis=0)

    var = centered.var().sort_values(ascending=False)
    genes = var.head(min(args.top_genes, len(var))).index.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(centered[genes].to_numpy(dtype=np.float32)).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) < 20:
        raise RuntimeError(f"Insufficient profiles ({len(X)}) for 100-epoch evaluation.")

    print(f"  Extracted profiles: {len(X)}")
    print(f"  Selected top variable genes: {len(genes)}")

    # Construct binary classification target for representation evaluation:
    # Perturbation Response Phenotype: High stress responder (top 50% shift magnitude) vs basal responder
    shift_magnitudes = np.linalg.norm(X, axis=1)
    median_shift = np.median(shift_magnitudes)
    y_class = (shift_magnitudes >= median_shift).astype(int)
    meta["perturbation_shift_magnitude"] = shift_magnitudes
    meta["is_high_responder"] = y_class

    pd.DataFrame(X, index=mat.index, columns=genes).rename_axis("profile_id").to_csv(OUT / "selected_feature_matrix.csv")
    meta.to_csv(OUT / "profile_metadata.csv", index=False)

    # Train / Validation Split (80 / 20)
    idx_all = np.arange(len(X))
    idx_train, idx_val = train_test_split(idx_all, test_size=args.val_ratio, random_state=args.seed, stratify=y_class)
    X_train, X_val = X[idx_train], X[idx_val]
    print(f"  Train set: {len(X_train)} profiles | Validation set: {len(X_val)} profiles (Stratified)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/6] Compute Device: {device} | Memory Tracking Active")

    d = X.shape[1]
    z = args.latent_dim

    # Instantiate all nine models
    models = {
        "TabNet": TabNetAE(d, z=z),
        "SAINT": SAINTAE(d, z=z, dim=32, heads=4, layers=2),
        "NODE": NODEAE(d, z=z),
        "GrowNet": GrowNetAE(d, z=z),
        "DANets": DANetAE(d, z=z),
        "ModernNCA": ModernNCAAE(d, z=z * 2),  # 32-dim contrastive metric space
        "VAE": VAE(d, z=z),
        "DBN": DBN(d, z=z),
        "DBM": DBM(d, z=z)
    }

    train_histories = {}
    val_histories = {}
    train_diagnostics = {}
    embeddings = {}
    reconstructions = {}

    print(f"\n[3/6] Executing 100-Epoch Training across all nine architectures...")

    for name, model in models.items():
        print(f"\n{'='*30} {name} (100 Epochs) {'='*30}")
        if name == "DBN":
            tr_h, val_h, t_time, mem = train_dbn_model(model, X_train, X_val, device, args.epochs, args.batch_size, args.lr)
        elif name == "ModernNCA":
            tr_h, val_h, t_time, mem = train_modernnca_model(model, X_train, X_val, device, args.epochs, args.batch_size, args.lr)
        else:
            tr_h, val_h, t_time, mem = train_reconstruction_model(name, model, X_train, X_val, device, args.epochs, args.batch_size, args.lr)

        train_histories[name] = tr_h
        val_histories[name] = val_h
        train_diagnostics[name] = {
            "training_time_sec": t_time,
            "final_train_loss": float(tr_h[-1]),
            "final_val_loss": float(val_h[-1]),
            **mem
        }

        # Save model weights
        torch.save(model.state_dict(), MODELS / f"{name.lower()}_state_dict.pt")

        # Save epoch-wise loss history
        loss_df = pd.DataFrame({
            "epoch": np.arange(1, len(tr_h) + 1),
            "train_loss": tr_h,
            "val_loss": val_h
        })
        loss_df.to_csv(OUT / f"{name.lower()}_loss.csv", index=False)

        # Plot individual model loss curve
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        ax.plot(loss_df["epoch"], loss_df["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2, linestyle="--")
        ax.set_title(f"{name}: 100-Epoch Loss Trajectory", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend()
        save_dual_figure(fig, FIG / f"{name.lower()}_loss")

        # Extract full dataset representation and reconstruction
        emb, recon = get_model_outputs(name, model, X, device)
        embeddings[name] = emb
        reconstructions[name] = recon

        print(f"  Completed {name}: Time = {t_time:.2f}s | Final Val Loss = {val_h[-1]:.6f} | Memory = {mem['summary']}")

    # 4. Compute All Metrics
    print("\n[4/6] Evaluating Classification, Representation, and Diagnostic Metrics...")

    classification_results = {}
    reconstruction_results = {}
    clustering_results = {}
    master_summary = []

    for name in models.keys():
        emb = embeddings[name]
        recon = reconstructions[name]

        # Classification metrics via 5-Fold Stratified CV
        cls_res = evaluate_classification_probe(emb, y_class, task_name=name, seed=args.seed)
        classification_results[name] = cls_res

        # Save individual confusion matrix and calibration curve
        cm_df = pd.DataFrame(cls_res["confusion_matrix_raw"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        cm_df.to_csv(OUT / f"{name.lower()}_confusion_matrix.csv")

        cal_df = pd.DataFrame({
            "mean_predicted_prob": cls_res["calibration"]["prob_pred"],
            "empirical_fraction_positive": cls_res["calibration"]["prob_true"]
        })
        cal_df.to_csv(OUT / f"{name.lower()}_calibration_curve.csv", index=False)

        # Reconstruction & Clustering metrics
        rep_res = evaluate_reconstruction_and_clustering(X, recon, emb, seed=args.seed)
        reconstruction_results[name] = rep_res

        # Save embeddings and clusters
        ecols = [f"{name}_z{i+1}" for i in range(emb.shape[1])]
        edf = pd.DataFrame(emb, columns=ecols)
        edf.insert(0, "profile_id", mat.index)
        edf = edf.merge(meta, on="profile_id", how="left")
        edf.to_csv(OUT / f"{name.lower()}_embeddings.csv", index=False)

        assign = meta.copy()
        assign["cluster"] = rep_res["cluster_labels"]
        assign.to_csv(OUT / f"{name.lower()}_clusters.csv", index=False)

        # PCA visualization
        pca = PCA(n_components=2, random_state=args.seed).fit_transform(emb)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for s in sorted(meta["sample_id"].astype(str).unique()):
            m = meta["sample_id"].astype(str).to_numpy() == s
            ax.scatter(pca[m, 0], pca[m, 1], label=s, alpha=0.8, edgecolors="none")
        ax.set_title(f"{name}: Latent PCA (Sample Source)", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        save_dual_figure(fig, FIG / f"{name.lower()}_pca")

        # Cluster visualization
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(pca[:, 0], pca[:, 1], c=rep_res["cluster_labels"], cmap="tab10", alpha=0.85)
        ax.set_title(f"{name}: Exploratory Clusters (k={rep_res['k_clusters']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="Cluster ID")
        ax.grid(True, alpha=0.2)
        save_dual_figure(fig, FIG / f"{name.lower()}_clusters")

        # Combine into master record
        diag = train_diagnostics[name]
        master_summary.append({
            "model": name,
            "epochs": args.epochs,
            "latent_dim": emb.shape[1],
            # Classification Metrics
            "auroc": cls_res["auroc"],
            "auprc": cls_res["auprc"],
            "accuracy": cls_res["accuracy"],
            "precision": cls_res["precision"],
            "recall": cls_res["recall"],
            "specificity": cls_res["specificity"],
            "f1_score": cls_res["f1_score"],
            "mcc": cls_res["mcc"],
            "brier_score": cls_res["brier_score"],
            "confusion_TN": cls_res["confusion_matrix"]["TN"],
            "confusion_FP": cls_res["confusion_matrix"]["FP"],
            "confusion_FN": cls_res["confusion_matrix"]["FN"],
            "confusion_TP": cls_res["confusion_matrix"]["TP"],
            # Reconstruction Metrics
            "mse": rep_res["mse"],
            "mae": rep_res["mae"],
            "rmse": rep_res["rmse"],
            "final_val_recon_loss": diag["final_val_loss"],
            # Clustering / Representation Metrics
            "silhouette_score": rep_res["silhouette_score"],
            "davies_bouldin_index": rep_res["davies_bouldin_index"],
            "calinski_harabasz_index": rep_res["calinski_harabasz_index"],
            # Diagnostics
            "final_train_loss": diag["final_train_loss"],
            "final_val_loss": diag["final_val_loss"],
            "training_time_sec": diag["training_time_sec"],
            "memory_usage": diag["summary"]
        })

    master_df = pd.DataFrame(master_summary)
    master_df.to_csv(OUT / "all_models_summary.csv", index=False)

    # Save dedicated metric CSVs
    cls_columns = ["model", "auroc", "auprc", "accuracy", "precision", "recall", "specificity", "f1_score", "mcc", "brier_score", "confusion_TN", "confusion_FP", "confusion_FN", "confusion_TP"]
    master_df[cls_columns].to_csv(OUT / "all_models_classification_metrics.csv", index=False)

    rec_columns = ["model", "mse", "mae", "rmse", "final_val_recon_loss", "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index"]
    master_df[rec_columns].to_csv(OUT / "all_models_reconstruction_metrics.csv", index=False)

    diag_columns = ["model", "epochs", "final_train_loss", "final_val_loss", "training_time_sec", "memory_usage"]
    master_df[diag_columns].to_csv(OUT / "all_models_diagnostics.csv", index=False)

    # 5. Generate Comprehensive Visualizations
    print("\n[5/6] Generating Publication-Quality Multi-Model Visualizations...")
    cls_dict_for_plots = {
        name: {
            "confusion_matrix_raw": classification_results[name]["cm_raw"],
            "accuracy": classification_results[name]["accuracy"],
            "f1_score": classification_results[name]["f1_score"],
            "mcc": classification_results[name]["mcc"],
            "brier_score": classification_results[name]["brier_score"],
            "auroc": classification_results[name]["auroc"],
            "auprc": classification_results[name]["auprc"],
            "calibration": classification_results[name]["calibration"],
            "roc": classification_results[name]["roc"],
            "pr": classification_results[name]["pr"]
        }
        for name in models.keys()
    }

    plot_all_confusion_matrices(cls_dict_for_plots, FIG)
    plot_all_calibration_curves(cls_dict_for_plots, FIG)
    plot_all_roc_and_pr_curves(cls_dict_for_plots, FIG)
    plot_training_validation_losses(train_histories, val_histories, FIG)
    plot_metric_barcharts(master_df, FIG)

    # 6. Final JSON Report
    print("\n[6/6] Writing Comprehensive Training & Benchmarking Report...")
    report = {
        "suite": "CRLP Nine-Model Complete Representation Learning Suite",
        "epochs_per_model": args.epochs,
        "total_models": len(models),
        "models": list(models.keys()),
        "profiles_count": len(X),
        "selected_gene_features": len(genes),
        "compute_device": str(device),
        "classification_task": "Perturbation Response Phenotype (High Stress Responder vs Basal Responder via 5-Fold Stratified CV Linear Probe)",
        "metrics_evaluated": {
            "classification": ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "MCC", "Brier score", "Confusion matrices", "Calibration curves"],
            "reconstruction_representation": ["MSE", "MAE", "RMSE", "Reconstruction loss", "Silhouette score", "Davies-Bouldin index", "Calinski-Harabasz index"],
            "diagnostics": ["Epoch-wise training loss", "Validation loss", "Training time per model", "GPU/RAM memory usage"]
        },
        "all_models_summary": master_summary
    }

    report_path = OUT / "all_models_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL NINE MODELS SUCCESSFULLY EXECUTED (100 EPOCHS EACH) END-TO-END!")
    print("=" * 80)
    print(master_df[["model", "auroc", "accuracy", "f1_score", "mse", "silhouette_score", "training_time_sec"]].to_string(index=False))
    print("=" * 80)
    print(f"Summary Table     : {OUT / 'all_models_summary.csv'}")
    print(f"Full Report JSON  : {report_path}")
    print(f"Figures Directory : {FIG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
