import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SeqDataset, VALID_PARQ
from model import DualGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = Path("checkpoints/best.pt")

def compute_wp(pred, y, mask):
    valid = mask > 0.5
    p = np.clip(pred[valid], -2, 2); yc = np.clip(y[valid], -2, 2)
    w = np.abs(yc); W = w.sum()
    if W < 1e-12: return 0.0
    pb = (w*p).sum()/W; yb = (w*yc).sum()/W
    cov = (w*(p-pb)*(yc-yb)).sum()/W
    vp  = (w*(p-pb)**2).sum()/W
    vy  = (w*(yc-yb)**2).sum()/W
    return float(cov/np.sqrt(vp*vy + 1e-12))

# --- NEW FILTERS ---
def causal_median(x, window):
    return pd.Series(x).rolling(window, min_periods=1).median().values

def causal_ema(x, alpha=0.1):
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().values

def quantize(x, step=0.5):
    return np.round(x / step) * step

print(f"Loading {CKPT_PATH}...")
model = DualGRU(hidden=64).to(DEVICE).eval()
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"], strict=False)
print(f"Loaded epoch {ckpt['epoch']}")

pf = pq.ParquetFile(str(VALID_PARQ))
seqs = []
for rg in range(3):
    df = pf.read_row_group(rg, columns=["seq_ix"]).to_pandas()
    seqs.extend(df["seq_ix"].unique().tolist())
plot_seqs = sorted(set(seqs))[:3]

all_seqs = []
for rg in range(20):
    df = pf.read_row_group(rg, columns=["seq_ix"]).to_pandas()
    all_seqs.extend(df["seq_ix"].unique().tolist())
eval_seqs = sorted(set(all_seqs))[:20]

plot_ds = SeqDataset(VALID_PARQ, plot_seqs)
eval_ds = SeqDataset(VALID_PARQ, eval_seqs)

print("\nEvaluating filters...")
P0, T0, P1, T1, M = [], [], [], [], []
with torch.no_grad():
    for i in range(len(eval_ds)):
        X, _, _, t0v, t1v, mask = eval_ds[i]
        X = X.unsqueeze(0).to(DEVICE)
        T = X.shape[1]
        p0_all = torch.zeros(T); p1_all = torch.zeros(T)
        state = None
        for c0 in range(0, T, 2000):
            c1 = min(c0 + 2000, T)
            out = model(X[:, c0:c1, :], state=state, use_ar=False)
            p0_all[c0:c1] = out["pred0"][0]; p1_all[c0:c1] = out["pred1"][0]
            state = out["state"]
        P0.append(p0_all.numpy()); P1.append(p1_all.numpy())
        T0.append(t0v.numpy()); T1.append(t1v.numpy())
        M.append(mask.numpy())

p0, p1 = np.concatenate(P0), np.concatenate(P1)
t0, t1, m = np.concatenate(T0), np.concatenate(T1), np.concatenate(M)

filters = {
    "Raw": lambda x: x,
    "MA(10)": lambda x: pd.Series(x).rolling(10, min_periods=1).mean().values,
    "EMA(0.1)": lambda x: causal_ema(x, 0.1),
    "Median(5)": lambda x: causal_median(x, 5),
    "Quantize(0.5)": lambda x: quantize(x, 0.5),
    "Med+Quant": lambda x: quantize(causal_median(x, 5), 0.5)
}

for name, f in filters.items():
    ma0 = f(p0)
    ma1 = f(p1)
    wp0 = compute_wp(ma0, t0, m)
    wp1 = compute_wp(ma1, t1, m)
    print(f"  {name:15s}: t0 WP={wp0:.4f} | t1 WP={wp1:.4f} | Avg={0.5*(wp0+wp1):.4f}")

# --- PLOT ---
print("\nGenerating plots...")
fig, axes = plt.subplots(len(plot_ds), 2, figsize=(16, 5*len(plot_ds)), sharex=True)
zoom = 5000

with torch.no_grad():
    for idx in range(len(plot_ds)):
        X, _, _, t0v, t1v, mask = plot_ds[idx]
        X = X.unsqueeze(0).to(DEVICE)
        T = X.shape[1]
        p0_all = torch.zeros(T); p1_all = torch.zeros(T)
        state = None
        for c0 in range(0, T, 2000):
            c1 = min(c0 + 2000, T)
            out = model(X[:, c0:c1, :], state=state, use_ar=False)
            p0_all[c0:c1] = out["pred0"][0]; p1_all[c0:c1] = out["pred1"][0]
            state = out["state"]
            
        raw0 = p0_all.numpy(); raw1 = p1_all.numpy()
        med0 = causal_median(raw0, 5); med1 = causal_median(raw1, 5)
        quant0 = quantize(med0, 0.5); quant1 = quantize(med1, 0.5)
        
        t = np.arange(zoom)
        
        ax_l = axes[idx, 0]
        ax_l.plot(t, t0v.numpy()[:zoom], color="black", lw=2, label="True")
        ax_l.plot(t, raw0[:zoom], color="tab:blue", lw=1, alpha=0.3, label="Raw")
        ax_l.plot(t, med0[:zoom], color="tab:orange", lw=1.5, label="Median(5)")
        ax_l.plot(t, quant0[:zoom], color="tab:red", lw=1.5, label="Med+Quant")
        ax_l.set_title(f"Seq {plot_ds.seq_ids[idx]} | t0"); ax_l.legend(loc="upper right")
        
        ax_r = axes[idx, 1]
        ax_r.plot(t, t1v.numpy()[:zoom], color="black", lw=2, label="True")
        ax_r.plot(t, raw1[:zoom], color="tab:green", lw=1, alpha=0.3, label="Raw")
        ax_r.plot(t, med1[:zoom], color="tab:orange", lw=1.5, label="Median(5)")
        ax_r.plot(t, quant1[:zoom], color="tab:purple", lw=1.5, label="Med+Quant")
        ax_r.set_title(f"Seq {plot_ds.seq_ids[idx]} | t1"); ax_r.legend(loc="upper right")

plt.tight_layout()
out_path = Path("plots") / "better_filters.png"
plt.savefig(out_path, dpi=110)
print(f"Saved plot to {out_path}")
