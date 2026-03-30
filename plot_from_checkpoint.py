"""
plot_from_checkpoint.py
=======================
Run this INSTEAD of retraining. It:
  1. Loads your saved best_model.pt
  2. Runs one forward pass on the test set
  3. Generates all 4 figures in ./figures/

Usage:
    python plot_from_checkpoint.py

Requirements:
  - best_model.pt must exist in the current directory
  - val_5M/ data folder must be accessible
  - All imports from lorentz_part_improved.py must be available
"""

import os
import subprocess

# ── GPU selection (must happen before torch import) ───────────────────────────
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
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Import everything from your main training file
from lorentz_part_improved import (
    load_and_preprocess, make_loaders,
    LorentzParT, get_biased_mask, CFG, DEVICE
)

# ── HARDCODED HISTORY from your actual run ────────────────────────────────────
# Paste in the numbers from your terminal output.
# SSL: every epoch logged (we only have every-5 so we interpolate the rest as None)
# Cls: every epoch logged

SSL_HISTORY = {
    'total': [1.1294, None, None, None, 0.0632, None, None, None, None, 0.0298,
              None, None, None, None, 0.0216, None, None, None, None, 0.0158,
              None, None, None, None, 0.0121, None, None, None, None, 0.0106,
              None, None, None, None, 0.0096, None, None, None, None, 0.0092,
              None, None, None, None, 0.0089, None, None, None, None, 0.0088],
    'mse':   [0.664, None, None, None, 0.052, None, None, None, None, 0.024,
              None, None, None, None, 0.017, None, None, None, None, 0.013,
              None, None, None, None, 0.009, None, None, None, None, 0.008,
              None, None, None, None, 0.007, None, None, None, None, 0.007,
              None, None, None, None, 0.006, None, None, None, None, 0.006],
    'phi':   [0.344, None, None, None, 0.020, None, None, None, None, 0.010,
              None, None, None, None, 0.009, None, None, None, None, 0.006,
              None, None, None, None, 0.005, None, None, None, None, 0.005,
              None, None, None, None, 0.005, None, None, None, None, 0.004,
              None, None, None, None, 0.004, None, None, None, None, 0.004],
    'mass':  [2.930, None, None, None, 0.017, None, None, None, None, 0.004,
              None, None, None, None, 0.007, None, None, None, None, 0.001,
              None, None, None, None, 0.001, None, None, None, None, 0.001,
              None, None, None, None, 0.001, None, None, None, None, 0.002,
              None, None, None, None, 0.002, None, None, None, None, 0.002],
}

# Classification: we have every-5 epochs up to 40 (where it was cut off)
# Fill known values; rest estimated by linear interpolation
CLS_HISTORY = {
    'train_acc': [
        0.2141, None, None, None,  # ep 1-4
        0.4001, None, None, None, None,  # ep 5-9
        0.4292, None, None, None, None,  # ep 10-14
        0.4791, None, None, None, None,  # ep 15-19
        0.5020, None, None, None, None,  # ep 20-24
        0.5222, None, None, None, None,  # ep 25-29
        0.5389, None, None, None, None,  # ep 30-34
        0.5469, None, None, None, None,  # ep 35-39
        0.5558,                           # ep 40  ← last known
    ],
    'val_acc': [
        0.2308, None, None, None,
        0.3798, None, None, None, None,
        0.4395, None, None, None, None,
        0.4909, None, None, None, None,
        0.5267, None, None, None, None,
        0.5398, None, None, None, None,
        0.5527, None, None, None, None,
        0.5465, None, None, None, None,
        0.5650,
    ],
}


def interpolate_history(hist_dict):
    """Fill None values with linear interpolation between known points."""
    result = {}
    for key, vals in hist_dict.items():
        arr = list(vals)
        # Find known indices
        known_idx = [i for i, v in enumerate(arr) if v is not None]
        if not known_idx:
            result[key] = arr
            continue
        for i in range(len(arr)):
            if arr[i] is None:
                # Find surrounding known values
                prev = max((k for k in known_idx if k < i), default=None)
                nxt  = min((k for k in known_idx if k > i), default=None)
                if prev is None:
                    arr[i] = arr[nxt]
                elif nxt is None:
                    arr[i] = arr[prev]
                else:
                    t = (i - prev) / (nxt - prev)
                    arr[i] = arr[prev] + t * (arr[nxt] - arr[prev])
        result[key] = arr
    return result


@torch.no_grad()
def collect_predictions(model, loader, cfg, device):
    """Collect classification + reconstruction predictions from test set."""
    model.eval()
    all_true, all_pred = [], []
    recon_true = {k: [] for k in ['pT', 'eta', 'phi', 'E']}
    recon_pred = {k: [] for k in ['pT', 'eta', 'phi', 'E']}
    feat_names = ['pT', 'eta', 'phi', 'E']

    print("  Collecting classification predictions …")
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, logits, _ = model(xb)
        all_true.extend(yb.argmax(1).cpu().numpy())
        all_pred.extend(logits.argmax(1).cpu().numpy())

        # Reconstruction pass
        mask_idx = get_biased_mask(xb.size(0), cfg['seq_len'], cfg['mask_ratio'], device)
        recon, _, _ = model(xb, mask_indices=mask_idx)

        B, num_mask = mask_idx.shape
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(mask_idx)
        pred_vals = recon[b_idx, mask_idx].cpu().numpy()        # (B, num_mask, 4)
        true_vals = xb.permute(0, 2, 1)[b_idx, mask_idx].cpu().numpy()  # (B, num_mask, 4)

        # ── PADDING FILTER ──────────────────────────────────────────────────
        # JetClass pads jets to 128 particles with zeros. These show up as a
        # sharp spike at the very left of pT/E distributions after normalisation.
        # Strategy: compute per-event particle validity mask using pT AND E.
        # A particle is real if its raw pT (index 0 before normalisation) > 0.
        # Post-normalisation: padding pT clusters far below real particles.
        # We use both pT and E channels: padding has BOTH near the same
        # normalised value. Use pT channel: exclude the bottom 8% if it forms
        # a clear spike (i.e. p8 < p15 - 0.5*std, meaning a gap).
        pt_flat = true_vals[:, :, 0].flatten()
        p8, p15 = np.percentile(pt_flat, 8), np.percentile(pt_flat, 15)
        threshold = p8 if (p15 - p8) > 0.15 else -np.inf  # only filter if gap exists

        valid = (true_vals[:, :, 0] > threshold)  # (B, num_mask)

        for i, name in enumerate(feat_names):
            recon_true[name].append(true_vals[:, :, i][valid].flatten())
            recon_pred[name].append(pred_vals[:, :, i][valid].flatten())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    for name in feat_names:
        recon_true[name] = np.concatenate(recon_true[name])
        recon_pred[name] = np.concatenate(recon_pred[name])

    print(f"  Collected {len(all_true)} test predictions.")
    return all_true, all_pred, recon_true, recon_pred


def plot_results(ssl_hist, cls_hist, all_true, all_pred, recon_true, recon_pred,
                 out_dir="figures"):
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
        'figure.dpi': 150,
    })

    # ── Figure 1: SSL Loss Convergence ─────────────────────────────────────────
    epochs_ssl = range(1, len(ssl_hist['total']) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs_ssl, ssl_hist['total'], color=NAVY,  lw=2.0, label='Total loss')
    ax.plot(epochs_ssl, ssl_hist['mse'],   color=BLUE,  lw=1.5, ls='--', label='MSE (pT, η, E)')
    ax.plot(epochs_ssl, ssl_hist['phi'],   color=TEAL,  lw=1.5, ls='--', label='Cyclic φ loss')
    ax.plot(epochs_ssl, ssl_hist['mass'],  color=AMBER, lw=1.5, ls='--', label='Invariant mass loss (new)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Phase 1 — SSL Pretraining Convergence (LorentzParT MAE)', fontweight='bold', color=NAVY)
    ax.legend(frameon=False)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = f"{out_dir}/ssl_loss_convergence.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✔ {p}")

    # ── Figure 2: Classification Accuracy ──────────────────────────────────────
    ep_cls = range(1, len(cls_hist['train_acc']) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep_cls, [a * 100 for a in cls_hist['train_acc']], color=BLUE, lw=2.0, label='Train accuracy')
    ax.plot(ep_cls, [a * 100 for a in cls_hist['val_acc']],   color=TEAL, lw=2.0, ls='--', label='Val accuracy')
    best_val = max(cls_hist['val_acc']) * 100
    best_ep  = cls_hist['val_acc'].index(max(cls_hist['val_acc'])) + 1
    ax.axhline(best_val, color=RED, lw=1.0, ls=':', alpha=0.7)
    ax.annotate(f'Best val: {best_val:.1f}%  (ep {best_ep})',
                xy=(best_ep, best_val), xytext=(best_ep + 1, best_val - 4.5),
                color=RED, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(15, 70)
    ax.set_title('Phase 2 — Classification Fine-Tuning (10-class JetClass)', fontweight='bold', color=NAVY)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)
    # Shade the epochs that were cut off (41-50) to show they weren't run
    ax.axvspan(40.5, len(ep_cls) + 0.5, alpha=0.07, color='gray', label='_nolegend_')
    ax.text(41, 20, 'training\ninterrupted', fontsize=8, color='gray', va='bottom')
    fig.tight_layout()
    p = f"{out_dir}/classification_accuracy.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✔ {p}")

    # ── Figure 3: Confusion Matrix ──────────────────────────────────────────────
    present = sorted(set(all_true.tolist()) | set(all_pred.tolist()))
    names   = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in present]
    cm = confusion_matrix(all_true, all_pred, labels=present, normalize='true')

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction of true class')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Predicted class', fontweight='bold')
    ax.set_ylabel('True class', fontweight='bold')
    ax.set_title('Confusion Matrix — LorentzParT Test Set (row-normalised)', fontweight='bold', color=NAVY)
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8 if len(names) <= 10 else 6,
                    color='white' if val > 0.55 else NAVY)
    fig.tight_layout()
    p = f"{out_dir}/confusion_matrix.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✔ {p}")

    # ── Figure 4: Reconstruction Histograms ────────────────────────────────────
    feat_info = {
        'pT':  'Normalised $p_T$  (log1p + z-score)',
        'eta': 'Normalised $\\eta$  (z-score)',
        'phi': '$\\phi$  (rad, not normalised)',
        'E':   'Normalised $E$  (log1p + z-score)',
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for idx, (name, xlabel) in enumerate(feat_info.items()):
        ax   = axes[idx]
        t    = recon_true[name]
        pr   = recon_pred[name]
        lo   = np.percentile(t, 1);  hi = np.percentile(t, 99)
        bins = np.linspace(lo, hi, 65)
        ax.hist(t,  bins=bins, histtype='stepfilled', alpha=0.35, color=BLUE,  label='True',      density=True)
        ax.hist(pr, bins=bins, histtype='step',       alpha=0.90, color=RED,   label='Predicted', density=True, lw=1.8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name} — masked particle reconstruction', fontweight='bold', color=NAVY, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Phase 1 — Masked Particle Reconstruction: True vs Predicted\n(evaluated on test set)',
                 fontweight='bold', fontsize=13, color=NAVY)
    fig.tight_layout()
    p = f"{out_dir}/reconstruction_histograms.png"
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {p}")

    print(f"\n✔ All 4 figures saved to ./{out_dir}/")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINT = "best_model.pt"

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"'{CHECKPOINT}' not found in current directory.\n"
            "Make sure you're running this from the same folder where training saved it."
        )

    # Load data (needed for test set predictions)
    print("[1/4] Loading data …")
    tensors = load_and_preprocess("val_5M", max_events=CFG['max_events'])
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = tensors
    print(f"      Test set size: {X_te.shape[0]}")
    _, _, te_loader = make_loaders(X_tr, Y_tr, X_va, Y_va, X_te, Y_te, CFG['batch_size'])

    # Load model
    print(f"[2/4] Loading checkpoint: {CHECKPOINT} …")
    model = LorentzParT(CFG).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Model parameters: {n_params:,}")

    # Quick accuracy check to confirm checkpoint is good
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, logits, _ = model(xb)
            correct += (logits.argmax(1) == yb.argmax(1)).sum().item()
            total   += len(yb)
    print(f"      Checkpoint test accuracy: {correct/total:.4f}  (expected ~0.565)")

    # Collect predictions
    print("[3/4] Collecting test set predictions …")
    all_true, all_pred, recon_true, recon_pred = collect_predictions(
        model, te_loader, CFG, DEVICE
    )

    # Interpolate history and plot
    print("[4/4] Generating figures …")
    ssl_hist = interpolate_history(SSL_HISTORY)
    cls_hist = interpolate_history(CLS_HISTORY)
    plot_results(ssl_hist, cls_hist, all_true, all_pred, recon_true, recon_pred,
                 out_dir="figures")
