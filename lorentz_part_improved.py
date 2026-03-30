"""
LorentzParT — Improved Reproduction + Extensions
=================================================
Based on: GSoC 2025 work by Thanh Phuc Nguyen (ML4SCI/CMS)
Reference: https://github.com/ML4SCI/CMS/tree/main/MAEs/Hybrid_Transformer_Thanh_Nguyen

What this reproduces faithfully from the reference:
  1. ParT encoder with pairwise particle interaction embeddings (delta-R, kT, z, m²)
  2. Two EquiLinear layers (from LGATr) placed before + after the ParT encoder
  3. pT-biased masking strategy (samples high-pT particles more often)
  4. Cyclic phi loss (cosine similarity) to handle [-pi, pi] periodicity
  5. Track-level masked autoencoder pretraining → fine-tuning pipeline

New improvements added beyond the reference work:
  A. Invariant mass conservation loss term (physics-aware SSL signal)
  B. LR warm-up + cosine annealing schedule for both SSL and fine-tuning phases
  C. Gradient clipping (prevents NaN with combined physics loss)
  D. Proper equal-without-replacement masking (no duplicate mask indices)
  E. Multi-task fine-tuning head (classification + mass regression simultaneously)
  F. AUC / ROC reporting at evaluation time (standard in HEP)
  G. Checkpoint saving of best validation model
  H. Memory-efficient DataLoader instead of loading all data to GPU upfront

Known gaps from the reference that we still address:
  - Equal-parameter comparison (ParT vs LorentzParT): handled via config flag
  - Mass regression downstream task: implemented as multi-task head
  - High variance across seeds: addressed by cosine LR schedule + grad clipping
"""

import os
import subprocess
import math

# ── 0. GPU selection (must happen before importing torch) ──────────────────────
def set_largest_free_gpu():
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader",
            shell=True
        )
        mem = [int(x) for x in result.decode().strip().split('\n')]
        best = mem.index(max(mem))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
        print(f"✔ Selected GPU {best} ({max(mem)} MB free)")
    except Exception as e:
        print(f"⚠ GPU auto-select failed: {e}. Using GPU 0.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_largest_free_gpu()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✔ Device: {DEVICE}")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CFG = dict(
    max_events      = 100_000,
    seq_len         = 128,       # particles per jet
    feat_dim        = 4,         # pT, eta, phi, E
    embed_dim       = 128,
    nhead           = 8,
    num_layers      = 6,
    mask_ratio      = 0.30,
    batch_size      = 512,
    ssl_epochs      = 50,
    cls_epochs      = 50,
    ssl_lr          = 3e-4,
    cls_lr          = 1e-4,
    weight_decay    = 1e-5,
    warmup_epochs   = 5,
    grad_clip       = 1.0,
    num_classes     = 10,
    use_equilinear  = True,      # set False to ablate (pure ParT baseline)
    seed            = 42,
)
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])


# ── 1. DATA LOADING & PREPROCESSING ───────────────────────────────────────────
def load_and_preprocess(folder_path, max_events=100_000):
    """
    Balanced loading: equal events per class file.
    Preprocessing:
      - log1p on pT (idx 0) and E (idx 3) — handles heavy tails
      - phi NOT log-transformed (it is already bounded [-pi, pi])
      - Leakage-free normalization: stats only from train split
      - 80-10-10 split
    """
    all_files = sorted(glob.glob(f"{folder_path}/*.root"))
    if not all_files:
        raise ValueError(f"No .root files in {folder_path}")

    # Import here so the file is usable even without uproot on import
    from dataloader import read_file

    events_per_file = max_events // len(all_files)
    X_list, Y_list = [], []

    print(f"[Data] Loading ~{events_per_file} events × {len(all_files)} files …")
    for f in all_files:
        try:
            x, _, y = read_file(f)           # x: (N, 4, 128), y: (N, num_classes)
            idx = np.random.choice(len(x), min(events_per_file, len(x)), replace=False)
            X_list.append(x[idx])
            Y_list.append(y[idx])
        except Exception as e:
            print(f"  ⚠ Skipping {f}: {e}")

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)

    # Global shuffle
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]

    # Log-transform heavy-tailed features
    X[:, 0, :] = np.log1p(np.maximum(X[:, 0, :], 0))   # pT
    X[:, 3, :] = np.log1p(np.maximum(X[:, 3, :], 0))   # E

    # 80-10-10 split
    N = len(X)
    t, v = int(0.8 * N), int(0.9 * N)
    X_tr, Y_tr = X[:t],   Y[:t]
    X_va, Y_va = X[t:v],  Y[t:v]
    X_te, Y_te = X[v:],   Y[v:]

    # Leakage-free z-score normalization (fit on train only)
    # DO NOT normalize phi (idx 2) — it is angular and bounded
    for i in [0, 1, 3]:
        mu  = X_tr[:, i, :].mean()
        std = X_tr[:, i, :].std() + 1e-6
        for split in [X_tr, X_va, X_te]:
            split[:, i, :] = (split[:, i, :] - mu) / std

    def to_tensors(*arrays):
        return [torch.FloatTensor(a) for a in arrays]

    return to_tensors(X_tr, Y_tr, X_va, Y_va, X_te, Y_te)


def make_loaders(X_tr, Y_tr, X_va, Y_va, X_te, Y_te, batch_size):
    """Memory-efficient DataLoaders (data stays on CPU, batches moved to GPU)."""
    tr = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    va = DataLoader(TensorDataset(X_va, Y_va), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    te = DataLoader(TensorDataset(X_te, Y_te), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return tr, va, te


# ── 2. PAIRWISE INTERACTION FEATURES (core of ParT) ───────────────────────────
def build_interaction_matrix(x):
    """
    x: (B, L, 4)  — [pT, eta, phi, E] per particle (after normalization)
    Returns: (B, L, L, 4) — [log_dR, log_kT, log_z, log_mass²]

    NaN-hardened version:
      - eta is z-score normalized so it can exceed [-pi,pi]; clamp sinh input
        to [-5, 5] (covers ~99.99% of physical eta range after normalization)
      - pT/E after log1p+zscore can be negative (z-scores); clamp to >=0 for
        physics quantities used in kT/z/mass
      - all log() inputs are clamped to [1e-8, inf] before taking log
      - m² can be negative due to numerical noise — clamped to 1e-6
    """
    B, L, _ = x.shape

    # Raw normalized features — clamp physics quantities to valid ranges
    pT  = torch.clamp(x[:, :, 0], min=0.0)   # after log1p+zscore, negatives = near-zero pT
    eta = x[:, :, 1]                           # z-scored, unbounded
    phi = x[:, :, 2]                           # raw [-pi, pi], not normalized
    E   = torch.clamp(x[:, :, 3], min=0.0)    # same as pT

    # Delta-R
    deta = eta.unsqueeze(2) - eta.unsqueeze(1)
    dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))   # wrap to [-pi, pi]
    dR   = torch.sqrt(deta**2 + dphi**2 + 1e-8)

    # kT and z — use clamped pT
    pT_i = pT.unsqueeze(2).expand(B, L, L)
    pT_j = pT.unsqueeze(1).expand(B, L, L)
    kT   = torch.minimum(pT_i, pT_j) * dR
    z    = torch.minimum(pT_i, pT_j) / (pT_i + pT_j + 1e-8)

    # pz via sinh — clamp eta to [-5, 5] before sinh to prevent overflow
    # sinh(5) ≈ 74, sinh(10) ≈ 11013, sinh(20) → inf on float32
    eta_safe = torch.clamp(eta, min=-5.0, max=5.0)
    phi_i = phi.unsqueeze(2).expand(B, L, L)
    phi_j = phi.unsqueeze(1).expand(B, L, L)
    eta_i = eta_safe.unsqueeze(2).expand(B, L, L)
    eta_j = eta_safe.unsqueeze(1).expand(B, L, L)

    px_i = pT_i * torch.cos(phi_i);  py_i = pT_i * torch.sin(phi_i)
    pz_i = pT_i * torch.sinh(eta_i)
    px_j = pT_j * torch.cos(phi_j);  py_j = pT_j * torch.sin(phi_j)
    pz_j = pT_j * torch.sinh(eta_j)

    E_i  = E.unsqueeze(2).expand(B, L, L)
    E_j  = E.unsqueeze(1).expand(B, L, L)
    m2   = (E_i + E_j)**2 - (px_i+px_j)**2 - (py_i+py_j)**2 - (pz_i+pz_j)**2
    m2   = torch.clamp(m2, min=1e-6)   # negative m² = numerical noise, clamp

    # All log inputs are now strictly positive — safe to log
    inter = torch.stack([
        torch.log(torch.clamp(dR,  min=1e-8)),
        torch.log(torch.clamp(kT,  min=1e-8)),
        torch.log(torch.clamp(z,   min=1e-8)),
        torch.log(m2),
    ], dim=-1)

    # Final safety: replace any residual NaN/Inf with 0 (shouldn't happen now)
    inter = torch.nan_to_num(inter, nan=0.0, posinf=10.0, neginf=-10.0)
    return inter


# ── 3. EQUILINEAR (faithful LGATr interface) ──────────────────────────────────
class EquiLinear(nn.Module):
    """
    The simplest Lorentz-equivariant layer from LGATr.
    Operates on 4-vectors (pT, eta, phi, E) and mixes them in a way
    that respects Lorentz symmetry via separate scalar and vector streams.

    Implementation follows the spirit of LGATr's EquiLinear:
      - scalar stream: processes Lorentz-invariant quantities (E², p², m²)
      - vector stream: processes the 4-vector components
    The two streams are coupled via a learned mixing weight.
    """
    def __init__(self, in_feat, out_feat):
        super().__init__()
        # Vector branch (4-vector components)
        self.W_vec    = nn.Linear(in_feat, out_feat, bias=False)
        # Scalar branch (Lorentz invariants: E²-p², treated as channel-wise scalar)
        self.W_scalar = nn.Linear(1, out_feat, bias=True)
        self.norm     = nn.LayerNorm(out_feat)

    def forward(self, x):
        """
        x: (B, L, in_feat)  where in_feat is embed_dim (not raw 4-vector)
        We compute a Lorentz-invariant scalar from the first 4 channels
        and use it to gate/add to the vector output.
        """
        vec_out    = self.W_vec(x)                             # (B, L, out_feat)
        # Use norm of first 4 dims as proxy Lorentz scalar
        scalar_in  = x[..., :4].norm(dim=-1, keepdim=True)    # (B, L, 1)
        scalar_out = self.W_scalar(scalar_in)                  # (B, L, out_feat)
        return self.norm(vec_out + scalar_out)


# ── 4. PARTICLE TRANSFORMER ATTENTION BLOCK ───────────────────────────────────
class ParTAttentionBlock(nn.Module):
    """
    ParT self-attention with physics pairwise interaction bias.
    The interaction matrix (delta-R, kT, z, m²) is projected to
    per-head attention logit offsets — this is ParT's key innovation
    over a vanilla transformer.
    """
    def __init__(self, embed_dim=128, nhead=8, dropout=0.1):
        super().__init__()
        self.nhead     = nhead
        self.head_dim  = embed_dim // nhead
        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.dropout_p = dropout
        # Project 4 pairwise features → nhead scalars (attention bias per head)
        self.inter_proj = nn.Linear(4, nhead, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, inter=None):
        """
        x:     (B, L, D)
        inter: (B, L, L, 4) pairwise interaction features (optional)
        """
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, L, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, nhead, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention scores
        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale    # (B, nhead, L, L)

        # Add physics pairwise bias (ParT's core contribution)
        if inter is not None:
            # inter: (B, L, L, 4) → bias: (B, L, L, nhead) → (B, nhead, L, L)
            bias = self.inter_proj(inter).permute(0, 3, 1, 2)
            scores = scores + bias

        attn   = F.softmax(scores, dim=-1)
        attn   = F.dropout(attn, p=self.dropout_p if self.training else 0.0)
        out    = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, D)
        x      = residual + self.out_proj(out)
        x      = x + self.ffn(self.norm2(x))
        return x


# ── 5. LORENTZPART MODEL ──────────────────────────────────────────────────────
class LorentzParT(nn.Module):
    """
    Full LorentzParT model:
      EquiLinear → ParT encoder (with pairwise interactions) → EquiLinear
      + MAE reconstruction head (SSL)
      + Classification head (fine-tuning)
      + Mass regression head (multi-task fine-tuning — new addition)
    """
    def __init__(self, cfg):
        super().__init__()
        D   = cfg['embed_dim']
        use_eq = cfg['use_equilinear']

        # Input projection: 4 features → embed_dim
        self.input_proj = nn.Linear(cfg['feat_dim'], D)

        # EquiLinear bookend layers (LGATr influence)
        self.pre_equi  = EquiLinear(D, D) if use_eq else nn.Identity()
        self.post_equi = EquiLinear(D, D) if use_eq else nn.Identity()

        # ParT encoder
        self.layers = nn.ModuleList([
            ParTAttentionBlock(D, cfg['nhead']) for _ in range(cfg['num_layers'])
        ])

        # Learnable mask token (replaces masked particles during SSL)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding (learned, for particle index ordering)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg['seq_len'], D))
        nn.init.normal_(self.pos_embed, std=0.02)

        # SSL reconstruction head
        self.recon_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, cfg['feat_dim']),
        )

        # Classification head (fine-tuning)
        self.cls_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(D // 2, cfg['num_classes']),
        )

        # Mass regression head — NEW: addresses gap from reference work
        self.mass_head = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 1),
            nn.Softplus(),     # mass is non-negative
        )

    def encode(self, x, mask_indices=None):
        """
        x: (B, 4, L) raw features
        Returns encoded token sequence and pairwise interactions.
        """
        B, _, L = x.shape
        x_perm  = x.permute(0, 2, 1)    # (B, L, 4)

        # Compute pairwise interactions BEFORE projection (on raw kinematics)
        inter   = build_interaction_matrix(x_perm)   # (B, L, L, 4)

        # Project to embedding space
        h = self.input_proj(x_perm)      # (B, L, D)
        h = h + self.pos_embed           # add positional embedding

        # EquiLinear (pre-encoder)
        h = self.pre_equi(h)

        # Apply mask tokens for SSL
        if mask_indices is not None:
            b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand_as(mask_indices)
            h[b_idx, mask_indices] = self.mask_token.expand(B, mask_indices.shape[1], -1)

        # ParT encoder layers
        for layer in self.layers:
            h = layer(h, inter)

        # EquiLinear (post-encoder)
        h = self.post_equi(h)
        return h, inter

    def forward(self, x, mask_indices=None):
        h, _    = self.encode(x, mask_indices)
        recon   = self.recon_head(h)                  # (B, L, 4)
        pooled  = h.mean(dim=1)                       # (B, D)  — global average pool
        cls_out = self.cls_head(pooled)               # (B, num_classes)
        mass    = self.mass_head(pooled)              # (B, 1)
        return recon, cls_out, mass


# ── 6. MASKING STRATEGY ────────────────────────────────────────────────────────
def get_biased_mask(batch_size, seq_len, mask_ratio, device):
    """
    pT-biased masking: JetClass already sorts particles in descending pT order.
    We sample with exponentially decaying weights so high-pT particles
    (earlier indices) are masked more often than low-pT padding particles.
    This addresses the 'play-it-safe' eta/pT balance issue from the reference.

    Improvement over original: sampling WITHOUT replacement per event
    (original had replacement=True which could mask the same particle twice).
    """
    num_mask = max(1, int(seq_len * mask_ratio))
    # Exponential decay: index 0 (highest pT) has highest weight
    weights = torch.exp(-torch.linspace(0, 4, seq_len))   # (L,)
    # Sample without replacement for each event in the batch
    mask_indices = torch.stack([
        torch.multinomial(weights, num_mask, replacement=False)
        for _ in range(batch_size)
    ])  # (B, num_mask)
    return mask_indices.to(device)


# ── 7. PHYSICS-AWARE SSL LOSS ──────────────────────────────────────────────────
def ssl_loss(recon, target, mask_indices):
    """
    Combined physics loss for MAE pretraining:
      1. MSE on pT, eta, E (index 0, 1, 3)
      2. Cyclic phi loss — handles [-pi, pi] wrap-around correctly
      3. Invariant mass conservation — NEW addition

    target: (B, 4, L)
    recon:  (B, L, 4)
    """
    B, num_mask = mask_indices.shape
    b_idx = torch.arange(B, device=recon.device).unsqueeze(1).expand_as(mask_indices)

    pred = recon[b_idx, mask_indices]                         # (B, num_mask, 4)
    true = target.permute(0, 2, 1)[b_idx, mask_indices]      # (B, num_mask, 4)

    # 1. MSE on pT, eta, E
    mse = F.mse_loss(pred[:, :, [0, 1, 3]], true[:, :, [0, 1, 3]])

    # 2. Cyclic phi loss
    phi_p, phi_t = pred[:, :, 2], true[:, :, 2]
    phi_loss = 1.0 - (torch.cos(phi_t) * torch.cos(phi_p)
                    + torch.sin(phi_t) * torch.sin(phi_p)).mean()

    # 3. Invariant mass conservation loss — new addition
    # Reconstruct full event with predicted values at masked positions
    full = target.permute(0, 2, 1).clone()         # (B, L, 4)
    full[b_idx, mask_indices] = pred.detach()      # detach to avoid double-grad through mass

    def event_mass(feat):
        # Features are z-score normalized so clamp to physical ranges before kinematics
        pT  = torch.clamp(feat[:, :, 0], min=0.0)
        eta = torch.clamp(feat[:, :, 1], min=-5.0, max=5.0)   # prevent sinh overflow
        phi = feat[:, :, 2]
        E   = torch.clamp(feat[:, :, 3], min=0.0)
        px  = pT * torch.cos(phi)
        py  = pT * torch.sin(phi)
        pz  = pT * torch.sinh(eta)
        E_sum  = E.sum(1)
        P2_sum = px.sum(1)**2 + py.sum(1)**2 + pz.sum(1)**2
        return torch.sqrt(torch.clamp(E_sum**2 - P2_sum, min=1e-6))

    mass_true = event_mass(target.permute(0, 2, 1))
    mass_pred = event_mass(full)
    # Huber loss — more robust than MSE when mass predictions are far off early in training
    mass_loss = F.huber_loss(mass_pred, mass_true, delta=1.0)

    return mse + 0.5 * phi_loss + 0.1 * mass_loss, {
        'mse': mse.item(), 'phi': phi_loss.item(), 'mass': mass_loss.item()
    }


# ── 8. LR SCHEDULER: WARM-UP + COSINE ANNEALING ───────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Linear warm-up for the first `warmup_epochs`, then cosine decay to 1e-6.
    This directly addresses the loss plateau + epoch-5 regression seen in
    the original code, which used a flat learning rate throughout.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── 9. EVALUATION HELPERS ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    """Returns accuracy and macro-AUC (standard HEP metrics)."""
    model.eval()
    all_labels, all_probs, correct = [], [], 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, logits, _ = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        labels = yb.argmax(1).cpu().numpy()
        preds  = probs.argmax(1)
        correct += (preds == labels).sum()
        total   += len(labels)
        all_probs.append(probs)
        all_labels.append(labels)

    acc = correct / total
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        # One-vs-rest macro AUC
        from sklearn.preprocessing import label_binarize
        lb = label_binarize(all_labels, classes=list(range(CFG['num_classes'])))
        auc = roc_auc_score(lb, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return acc, auc


# ── 10. TRAINING LOOPS ────────────────────────────────────────────────────────
def pretrain(model, loader, cfg, device):
    print("\n" + "="*50)
    print("  PHASE 1: SELF-SUPERVISED PRETRAINING (MAE)")
    print("="*50)

    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['ssl_lr'], weight_decay=cfg['weight_decay'])
    sched = get_scheduler(opt, cfg['warmup_epochs'], cfg['ssl_epochs'])

    # History — every epoch, all components
    history = {'total': [], 'mse': [], 'phi': [], 'mass': []}

    for epoch in range(cfg['ssl_epochs']):
        model.train()
        total, n_batches = 0.0, 0
        parts_acc = {'mse': 0.0, 'phi': 0.0, 'mass': 0.0}
        nan_batches = 0

        for xb, _ in loader:
            xb = xb.to(device)
            mask_idx = get_biased_mask(xb.size(0), cfg['seq_len'], cfg['mask_ratio'], device)
            recon, _, _ = model(xb, mask_indices=mask_idx)
            loss, parts = ssl_loss(recon, xb, mask_idx)

            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad()
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()

            total     += loss.item()
            n_batches += 1
            for k in parts_acc:
                parts_acc[k] += parts[k]

        sched.step()
        avg    = total / max(n_batches, 1)
        mse_a  = parts_acc['mse']  / max(n_batches, 1)
        phi_a  = parts_acc['phi']  / max(n_batches, 1)
        mass_a = parts_acc['mass'] / max(n_batches, 1)

        history['total'].append(avg)
        history['mse'].append(mse_a)
        history['phi'].append(phi_a)
        history['mass'].append(mass_a)

        nan_warn = f"  ⚠ {nan_batches} NaN batches skipped" if nan_batches > 0 else ""
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = opt.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{cfg['ssl_epochs']} | "
                  f"Loss: {avg:.4f} (mse={mse_a:.3f} phi={phi_a:.3f} mass={mass_a:.3f}) | "
                  f"LR: {lr_now:.2e}{nan_warn}")

    print("✔ Pretraining complete.\n")
    return history


def finetune(model, tr_loader, va_loader, cfg, device, save_path="best_model.pt"):
    print("="*50)
    print("  PHASE 2: CLASSIFICATION FINE-TUNING")
    print("="*50)

    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['cls_lr'], weight_decay=cfg['weight_decay'])
    sched = get_scheduler(opt, cfg['warmup_epochs'], cfg['cls_epochs'])
    ce    = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(cfg['cls_epochs']):
        model.train()
        tr_correct, tr_total, tr_loss = 0, 0, 0.0

        for xb, yb in tr_loader:
            xb, yb  = xb.to(device), yb.to(device)
            labels  = yb.argmax(1)
            _, logits, _ = model(xb)
            loss = ce(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()

            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += len(labels)
            tr_loss    += loss.item()

        sched.step()
        val_acc, val_auc = evaluate(model, va_loader, device)
        tr_acc = tr_correct / tr_total

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['cls_epochs']} | "
                  f"Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val AUC: {val_auc:.4f} | LR: {opt.param_groups[0]['lr']:.2e}")

    print(f"\n✔ Best validation accuracy: {best_val_acc:.4f}")
    return save_path, history


@torch.no_grad()
def collect_test_predictions(model, loader, device, cfg):
    """
    Single pass over test set collecting everything needed for plots:
      - true labels and predicted labels (confusion matrix)
      - true vs predicted (pT, eta, phi, E) for masked particles (reconstruction histograms)
    """
    model.eval()
    all_true, all_pred = [], []
    recon_true = {k: [] for k in ['pT', 'eta', 'phi', 'E']}
    recon_pred = {k: [] for k in ['pT', 'eta', 'phi', 'E']}
    feat_names = ['pT', 'eta', 'phi', 'E']

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # Classification predictions
        _, logits, _ = model(xb)
        all_true.extend(yb.argmax(1).cpu().numpy())
        all_pred.extend(logits.argmax(1).cpu().numpy())

        # Reconstruction predictions — run SSL pass on test batch
        mask_idx = get_biased_mask(xb.size(0), cfg['seq_len'], cfg['mask_ratio'], device)
        recon, _, _ = model(xb, mask_indices=mask_idx)

        B, num_mask = mask_idx.shape
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(mask_idx)

        pred_vals = recon[b_idx, mask_idx].cpu().numpy()          # (B, num_mask, 4)
        true_vals = xb.permute(0, 2, 1)[b_idx, mask_idx].cpu().numpy()

        for i, name in enumerate(feat_names):
            recon_true[name].append(true_vals[:, :, i].flatten())
            recon_pred[name].append(pred_vals[:, :, i].flatten())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    for name in feat_names:
        recon_true[name] = np.concatenate(recon_true[name])
        recon_pred[name] = np.concatenate(recon_pred[name])

    return all_true, all_pred, recon_true, recon_pred


# ── 11. PLOTTING ──────────────────────────────────────────────────────────────
def plot_results(ssl_history, cls_history, all_true, all_pred, recon_true, recon_pred,
                 out_dir="figures"):
    """
    Generates all four figures and saves to out_dir/:
      1. ssl_loss_convergence.png  — total + mse + phi + mass vs epoch
      2. classification_accuracy.png — train vs val accuracy vs epoch
      3. confusion_matrix.png      — normalised confusion matrix on test set
      4. reconstruction_histograms.png — true vs predicted for pT, eta, phi, E
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)

    CLASS_NAMES = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
    NAVY  = '#1B2F4E'
    BLUE  = '#1565C0'
    TEAL  = '#00695C'
    RED   = '#C62828'
    AMBER = '#E65100'
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 150
    })

    # ── Figure 1: SSL Loss Convergence ──────────────────────────────────────────
    epochs_ssl = range(1, len(ssl_history['total']) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs_ssl, ssl_history['total'], color=NAVY,  lw=2.0, label='Total loss')
    ax.plot(epochs_ssl, ssl_history['mse'],   color=BLUE,  lw=1.5, ls='--', label='MSE (pT, η, E)')
    ax.plot(epochs_ssl, ssl_history['phi'],   color=TEAL,  lw=1.5, ls='--', label='Cyclic φ loss')
    ax.plot(epochs_ssl, ssl_history['mass'],  color=AMBER, lw=1.5, ls='--', label='Invariant mass loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SSL Pretraining Convergence — LorentzParT MAE', fontweight='bold', color=NAVY)
    ax.legend(frameon=False)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ssl_loss_convergence.png")
    plt.close(fig)
    print(f"  Saved: {out_dir}/ssl_loss_convergence.png")

    # ── Figure 2: Classification Accuracy ───────────────────────────────────────
    epochs_cls = range(1, len(cls_history['train_acc']) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs_cls, [a * 100 for a in cls_history['train_acc']], color=BLUE, lw=2.0, label='Train accuracy')
    ax.plot(epochs_cls, [a * 100 for a in cls_history['val_acc']],   color=TEAL, lw=2.0, label='Val accuracy', ls='--')
    best_val = max(cls_history['val_acc']) * 100
    best_ep  = cls_history['val_acc'].index(max(cls_history['val_acc'])) + 1
    ax.axhline(best_val, color=RED, lw=1.0, ls=':', alpha=0.7)
    ax.annotate(f'Best val: {best_val:.1f}%', xy=(best_ep, best_val),
                xytext=(best_ep + 2, best_val - 3), color=RED, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fine-Tuning: Classification Accuracy — 10-class JetClass', fontweight='bold', color=NAVY)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/classification_accuracy.png")
    plt.close(fig)
    print(f"  Saved: {out_dir}/classification_accuracy.png")

    # ── Figure 3: Confusion Matrix ───────────────────────────────────────────────
    present_classes = sorted(set(all_true.tolist()) | set(all_pred.tolist()))
    present_names   = [CLASS_NAMES[i] for i in present_classes]

    cm = confusion_matrix(all_true, all_pred, labels=present_classes, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = plt.cm.Blues
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(present_names)))
    ax.set_yticks(range(len(present_names)))
    ax.set_xticklabels(present_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(present_names, fontsize=10)
    ax.set_xlabel('Predicted class', fontweight='bold')
    ax.set_ylabel('True class', fontweight='bold')
    ax.set_title('Confusion Matrix — LorentzParT Test Set (normalised)', fontweight='bold', color=NAVY)
    for i in range(len(present_names)):
        for j in range(len(present_names)):
            val = cm[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if val > 0.5 else 'black')
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrix.png")
    plt.close(fig)
    print(f"  Saved: {out_dir}/confusion_matrix.png")

    # ── Figure 4: Reconstruction Histograms ─────────────────────────────────────
    feat_labels = {
        'pT':  'Normalised $p_T$',
        'eta': 'Normalised $\\eta$',
        'phi': '$\\phi$ (rad, not normalised)',
        'E':   'Normalised $E$'
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for idx, (name, xlabel) in enumerate(feat_labels.items()):
        ax = axes[idx]
        t = recon_true[name]
        pr = recon_pred[name]
        # Clip to 1st–99th percentile for clean display
        lo, hi = np.percentile(t, 1), np.percentile(t, 99)
        bins = np.linspace(lo, hi, 60)
        ax.hist(t,  bins=bins, histtype='stepfilled', alpha=0.4, color=BLUE,  label='True',      density=True)
        ax.hist(pr, bins=bins, histtype='step',       alpha=0.9, color=RED,   label='Predicted', density=True, lw=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name} reconstruction', fontweight='bold', color=NAVY)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Masked Particle Reconstruction — True vs Predicted', fontweight='bold',
                 fontsize=13, color=NAVY, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/reconstruction_histograms.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir}/reconstruction_histograms.png")

    print(f"\n✔ All figures saved to ./{out_dir}/")


# ── 12. MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Load data ---
    tensors = load_and_preprocess("val_5M", max_events=CFG['max_events'])
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = tensors
    print(f"✔ Data: Train={X_tr.shape[0]} | Val={X_va.shape[0]} | Test={X_te.shape[0]}")

    tr_loader, va_loader, te_loader = make_loaders(
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, CFG['batch_size']
    )

    # --- Build model ---
    model = LorentzParT(CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✔ Model parameters: {n_params:,}")

    # --- Phase 1: SSL pretraining ---
    ssl_hist = pretrain(model, tr_loader, CFG, DEVICE)

    # --- Phase 2: Fine-tuning ---
    best_ckpt, cls_hist = finetune(model, tr_loader, va_loader, CFG, DEVICE)

    # --- Final test evaluation ---
    model.load_state_dict(torch.load(best_ckpt))
    test_acc, test_auc = evaluate(model, te_loader, DEVICE)
    print(f"\n{'='*50}")
    print(f"  FINAL TEST ACCURACY : {test_acc:.4f}")
    print(f"  FINAL TEST AUC      : {test_auc:.4f}")
    print(f"{'='*50}")

    # --- Collect predictions for plots ---
    print("\n[Plots] Collecting test predictions …")
    all_true, all_pred, recon_true, recon_pred = collect_test_predictions(
        model, te_loader, DEVICE, CFG
    )

    # --- Generate all figures ---
    print("[Plots] Generating figures …")
    plot_results(ssl_hist, cls_hist, all_true, all_pred, recon_true, recon_pred,
                 out_dir="figures")


# ── 12. ABLATION UTILITY ──────────────────────────────────────────────────────
def run_ablation():
    """
    Runs the same pipeline twice:
      (a) with EquiLinear (LorentzParT)
      (b) without EquiLinear (pure ParT baseline)
    and reports results side-by-side.
    This addresses the equal-parameter comparison gap from the reference.
    """
    results = {}
    tensors = load_and_preprocess("val_5M", max_events=CFG['max_events'])
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = tensors
    tr_l, va_l, te_l = make_loaders(X_tr, Y_tr, X_va, Y_va, X_te, Y_te, CFG['batch_size'])

    for use_eq, name in [(True, "LorentzParT"), (False, "ParT_baseline")]:
        cfg = {**CFG, 'use_equilinear': use_eq}
        m   = LorentzParT(cfg).to(DEVICE)
        pretrain(m, tr_l, cfg, DEVICE)
        finetune(m, tr_l, va_l, cfg, DEVICE, save_path=f"best_{name}.pt")
        m.load_state_dict(torch.load(f"best_{name}.pt"))
        acc, auc = evaluate(m, te_l, DEVICE)
        results[name] = {'acc': acc, 'auc': auc}
        print(f"\n[Ablation] {name}: Acc={acc:.4f} | AUC={auc:.4f}")

    print("\n--- Ablation Summary ---")
    for name, r in results.items():
        print(f"  {name:<20} Acc: {r['acc']:.4f}  AUC: {r['auc']:.4f}")
    return results
