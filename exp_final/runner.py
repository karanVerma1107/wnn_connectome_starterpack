"""10-experiment sweep. 336 features (raw+m100+s100). 2000 train / 500 valid / 2 epochs.
Each experiment logged separately. Failures don't stop the sweep."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json, time, random, gc, traceback
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("/mnt/data/wnn_connectome_starterpack")
TRAIN_PARQ = ROOT / "wnn_connectome_starterpack/train_labeled.parquet"
VALID_PARQ = ROOT / "datasets/valid_labeled.parquet"
STATS_PATH = ROOT / "exp_sweep/stats_112.json"

FEATURES = (
    [f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)] +
    [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)] +
    [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)] +
    [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)] +
    [f"a{i}" for i in range(8)]
)
with open(STATS_PATH) as f: _s = json.load(f)
MU = np.array(_s["mean"], np.float32)
SIG = np.array(_s["std"], np.float32); SIG = np.where(SIG < 1e-8, 1.0, SIG)

# ---------- FEATURES: 336 dims = 112 raw + 112 m100 + 112 s100 ----------
def get_rolling(x, W):
    T, F = x.shape
    xp = np.concatenate([np.repeat(x[:1], W-1, axis=0), x], 0).astype(np.float64)
    cs = np.cumsum(xp, 0); csq = np.cumsum(xp*xp, 0)
    es = cs[W-1:]; esq = csq[W-1:]
    ss = np.concatenate([np.zeros((1,F)), cs[:-W]], 0)
    ssq = np.concatenate([np.zeros((1,F)), csq[:-W]], 0)
    m = (es-ss)/W
    v = np.maximum((esq-ssq)/W - m*m, 0)
    return m.astype(np.float32), np.sqrt(v).astype(np.float32)

def build_features(X_raw):
    X = (X_raw - MU) / SIG
    m100, s100 = get_rolling(X, 100)
    return np.concatenate([X, m100, s100], axis=-1).astype(np.float32)

def make_cls(t, W=20, thresh=0.3):
    T = len(t); c = np.zeros(T, dtype=np.int64)
    if T > W:
        d = t[W:] - t[:-W]
        c[:T-W] = np.where(d > thresh, 1, np.where(d < -thresh, 2, 0))
    return c

class SeqDataset(Dataset):
    def __init__(self, parquet_path, seq_ids):
        self.parquet_path = str(parquet_path); self.seq_ids = seq_ids
        self.pf = pq.ParquetFile(self.parquet_path); self.seq_to_rg = {}
        self.has_is_scored = "is_scored" in self.pf.schema.names
        for rg in range(self.pf.num_row_groups):
            df = self.pf.read_row_group(rg, columns=["seq_ix"]).to_pandas()
            for sid in df["seq_ix"].unique(): self.seq_to_rg[int(sid)] = rg
    def __len__(self): return len(self.seq_ids)
    def __getitem__(self, idx):
        sid = self.seq_ids[idx]; rg = self.seq_to_rg[sid]
        cols = ["seq_ix", "step_in_seq", "t0", "t1"] + FEATURES
        if self.has_is_scored: cols.append("is_scored")
        df = self.pf.read_row_group(rg, columns=cols).to_pandas()
        df = df[df["seq_ix"] == sid].sort_values("step_in_seq")
        X = build_features(df[FEATURES].to_numpy(np.float32))
        t0 = df["t0"].to_numpy(np.float32).copy(); t1 = df["t1"].to_numpy(np.float32).copy()
        mask = df["is_scored"].to_numpy(np.float32) if self.has_is_scored else np.ones(len(df), np.float32)
        return (torch.from_numpy(X), torch.from_numpy(make_cls(t0)), torch.from_numpy(make_cls(t1)),
                torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(mask))

# ---------- LOSS ----------
def wp_loss(pred, target):
    y = target.clamp(-2,2); w = y.abs(); p = pred.clamp(-2,2)
    W = w.sum()+1e-6; pm=(w*p).sum()/W; ym=(w*y).sum()/W
    cov=(w*(p-pm)*(y-ym)).sum()/W; vp=(w*(p-pm)**2).sum()/W; vy=(w*(y-ym)**2).sum()/W
    return -cov/torch.sqrt(vp*vy+1e-6)

class FocalLoss(nn.Module):
    def __init__(self, g=2.0): super().__init__(); self.g=g
    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction='none', ignore_index=-1)
        pt = torch.exp(-ce); return (((1-pt)**self.g)*ce).mean()

def total_loss(out, y0c, y1c, t0v, t1v, focal, mask=None):
    ce = focal(out["l0"].reshape(-1,3), y0c.reshape(-1)) + focal(out["l1"].reshape(-1,3), y1c.reshape(-1))
    wp = wp_loss(out["pred0"], t0v) + wp_loss(out["pred1"], t1v)
    gate_pen = (out["p_plat0"]*(1-out["p_plat0"])).mean() + (out["p_plat1"]*(1-out["p_plat1"])).mean()
    mse = torch.tensor(0.0, device=out["pred0"].device)
    if mask is not None:
        m = mask > 0.5
        if m.sum() > 0:
            mse = F.mse_loss(out["pred0"][m], t0v[m]) + F.mse_loss(out["pred1"][m], t1v[m])
    return ce + 0.3*wp + 0.5*mse + 0.1*gate_pen

# ---------- MODELS ----------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__(); self.pad=(k-1)*d
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        h = x.transpose(1,2)
        hc = F.relu(self.conv(F.pad(h,(self.pad,0))))
        hr = self.res(h) if isinstance(self.res, nn.Conv1d) else h
        return (hc + hr).transpose(1,2)

class TCN(nn.Module):
    def __init__(self, in_f, ch=48, k=3):
        super().__init__(); self.k=k; self.dils=[1,4,16,64]
        self.convs = nn.ModuleList(); prev = in_f
        for d in self.dils:
            self.convs.append(TCNBlock(prev, ch, k, d)); prev = ch
        self.out_ch = ch
        self.receptive_field = 1 + sum((k-1)*d for d in self.dils)
    def forward(self, x):
        for c in self.convs: x = c(x)
        return x

def make_heads(h, n_ens=3):
    return nn.ModuleList([nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 10)) for _ in range(n_ens)])

def decode_heads(heads, h):
    head = sum(hd(h) for hd in heads) / len(heads)
    p0 = F.softmax(head[..., 0:3], -1); p1 = F.softmax(head[..., 3:6], -1)
    d0 = torch.tanh(3.0 * (p0[...,1]-p0[...,2])); d1 = torch.tanh(3.0 * (p1[...,1]-p1[...,2]))
    feat0 = head[...,6] + d0 * F.softplus(head[...,8])
    feat1 = head[...,7] + d1 * F.softplus(head[...,9])
    return {"l0": head[...,0:3], "l1": head[...,3:6],
            "pred0": feat0, "pred1": feat1,
            "p_plat0": p0[...,0], "p_plat1": p1[...,0]}

class SingleTCN_GRU(nn.Module):
    """#9/#11: single TCN → GRU → 3 heads. 336 feat, 48ch."""
    def __init__(self, in_f=336, tcn_ch=48, gru_h=128):
        super().__init__(); self.tcn = TCN(in_f, tcn_ch); self.gru = nn.GRU(tcn_ch, gru_h, batch_first=True)
        self.heads = make_heads(gru_h); self.receptive_field = self.tcn.receptive_field
    def forward(self, x, state=None, use_ar=False):
        h = self.tcn(x); out, _ = self.gru(h); return decode_heads(self.heads, out)

class DualTCN_GRU(nn.Module):
    """#2/#6: dual TCN + gated fusion + GRU. x0=x1=same 336 feat (both branches)."""
    def __init__(self, in_f=336, tcn_ch=48, gru_h=128):
        super().__init__(); self.tcn0 = TCN(in_f, tcn_ch); self.tcn1 = TCN(in_f, tcn_ch)
        self.gate = nn.Sequential(nn.Linear(tcn_ch*2, tcn_ch), nn.Sigmoid())
        self.P1 = nn.Linear(tcn_ch, tcn_ch, bias=False)
        self.gru = nn.GRU(tcn_ch, gru_h, batch_first=True); self.heads = make_heads(gru_h)
        self.receptive_field = self.tcn0.receptive_field
    def forward(self, x, state=None, use_ar=False):
        h0 = self.tcn0(x); h1 = self.tcn1(x)
        g = self.gate(torch.cat([h0,h1], -1)); hf = h0 + g * self.P1(h1)
        out, _ = self.gru(hf); return decode_heads(self.heads, out)

class ParallelTCN_GRU(nn.Module):
    """#1: TCN and GRU run IN PARALLEL on same input → gate → heads."""
    def __init__(self, in_f=336, tcn_ch=48, gru_h=128):
        super().__init__(); self.tcn = TCN(in_f, tcn_ch); self.gru = nn.GRU(in_f, gru_h, batch_first=True)
        # Project both to common dim for gating
        self.proj_tcn = nn.Linear(tcn_ch, gru_h)
        self.proj_gru = nn.Linear(gru_h, gru_h)
        self.gate = nn.Sequential(nn.Linear(gru_h*2, gru_h), nn.Sigmoid())
        self.heads = make_heads(gru_h); self.receptive_field = self.tcn.receptive_field
    def forward(self, x, state=None, use_ar=False):
        t = self.proj_tcn(self.tcn(x))         # (B,T,H)
        g_out, _ = self.gru(x)                  # GRU sees raw features directly
        g_out = self.proj_gru(g_out)            # (B,T,H)
        gate = self.gate(torch.cat([t, g_out], -1))
        fused = t * gate + g_out * (1 - gate)
        return decode_heads(self.heads, fused)

class DualGRU_Gate(nn.Module):
    """#4: two GRUs (one per stream of same 336 feat) → gate → heads. Explicit fusion."""
    def __init__(self, in_f=336, gru_h=64):
        super().__init__(); self.gru0 = nn.GRU(in_f, gru_h, batch_first=True); self.gru1 = nn.GRU(in_f, gru_h, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(gru_h*2, gru_h), nn.Sigmoid())
        self.P1 = nn.Linear(gru_h, gru_h, bias=False); self.heads = make_heads(gru_h)
        self.receptive_field = 1
    def forward(self, x, state=None, use_ar=False):
        o0, _ = self.gru0(x); o1, _ = self.gru1(x)
        g = self.gate(torch.cat([o0,o1], -1)); hf = o0 + g * self.P1(o1)
        return decode_heads(self.heads, hf)

class ParallelTCN_GRU_Concat(nn.Module):
    """#5: TCN + GRU parallel, concat (no gate)."""
    def __init__(self, in_f=336, tcn_ch=48, gru_h=128):
        super().__init__(); self.tcn = TCN(in_f, tcn_ch); self.gru = nn.GRU(in_f, gru_h, batch_first=True)
        self.heads = make_heads(tcn_ch + gru_h); self.receptive_field = self.tcn.receptive_field
    def forward(self, x, state=None, use_ar=False):
        t = self.tcn(x); g_out, _ = self.gru(x)
        fused = torch.cat([t, g_out], -1)
        return decode_heads(self.heads, fused)

class SingleGRU(nn.Module):
    """#5': pure GRU h128, no TCN."""
    def __init__(self, in_f=336, gru_h=128):
        super().__init__(); self.gru = nn.GRU(in_f, gru_h, batch_first=True)
        self.heads = make_heads(gru_h); self.receptive_field = 1
    def forward(self, x, state=None, use_ar=False):
        out, _ = self.gru(x); return decode_heads(self.heads, out)

class DualTCN_DualGRU(nn.Module):
    """#3: max capacity - 2 TCNs + 2 GRUs + gate."""
    def __init__(self, in_f=336, tcn_ch=48, gru_h=64):
        super().__init__(); self.tcn0=TCN(in_f,tcn_ch); self.tcn1=TCN(in_f,tcn_ch)
        self.gru0=nn.GRU(tcn_ch, gru_h, batch_first=True); self.gru1=nn.GRU(tcn_ch, gru_h, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(gru_h*2, gru_h), nn.Sigmoid())
        self.P1 = nn.Linear(gru_h, gru_h, bias=False); self.heads = make_heads(gru_h)
        self.receptive_field = self.tcn0.receptive_field
    def forward(self, x, state=None, use_ar=False):
        h0, _ = self.gru0(self.tcn0(x)); h1, _ = self.gru1(self.tcn1(x))
        g = self.gate(torch.cat([h0,h1], -1)); hf = h0 + g * self.P1(h1)
        return decode_heads(self.heads, hf)

# ---------- TRAINING ----------
def evaluate(model, ds, DEV):
    model.eval(); P0,T0,P1,T1,M = [],[],[],[],[]
    with torch.no_grad():
        for X, _, _, t0v, t1v, mask in DataLoader(ds, batch_size=2, shuffle=False):
            X = X.to(DEV); out = model(X, state=None, use_ar=False)
            P0.append(out["pred0"].cpu().numpy().ravel()); P1.append(out["pred1"].cpu().numpy().ravel())
            T0.append(t0v.numpy().ravel()); T1.append(t1v.numpy().ravel()); M.append(mask.numpy().ravel())
    p0,t0,p1,t1,m = [np.concatenate(x) for x in (P0,T0,P1,T1,M)]
    def wp(p, y, m):
        v = m > 0.5; p = np.clip(p[v],-2,2); yc = np.clip(y[v],-2,2)
        w = np.abs(yc); W = w.sum()
        if W < 1e-12: return 0.0
        pb=(w*p).sum()/W; yb=(w*yc).sum()/W
        cov=(w*(p-pb)*(yc-yb)).sum()/W; vp=(w*(p-pb)**2).sum()/W; vy=(w*(yc-yb)**2).sum()/W
        return float(cov/np.sqrt(vp*vy+1e-12))
    wp0, wp1 = wp(p0,t0,m), wp(p1,t1,m)
    return 0.5*(wp0+wp1), wp0, wp1

def run_experiment(name, ModelClass, model_kwargs, train_ds, valid_ds, DEV, epochs=2):
    print(f"\n{'='*70}\n[EXP] {name}\n{'='*70}", flush=True)
    t_start = time.time()
    model = ModelClass(**model_kwargs).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}  RF: {model.receptive_field}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    focal = FocalLoss()
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    best_wp = -1.0
    for epoch in range(1, epochs+1):
        model.train(); el = 0.0; n_batch = 0
        for bi, (X, y0c, y1c, t0v, t1v, mask_b) in enumerate(train_loader):
            X,y0c,y1c,t0v,t1v,mask_b = [z.to(DEV) for z in (X,y0c,y1c,t0v,t1v,mask_b)]
            opt.zero_grad(set_to_none=True)
            try:
                out = model(X, state=None, use_ar=False)
            except Exception as e:
                print(f"  [forward fail] {e}"); return {"name": name, "error": str(e)}
            # Trim warmup rows
            out_s = {k: out[k][:, 99:] for k in ["l0","l1","pred0","pred1","p_plat0","p_plat1"]}
            loss = total_loss(out_s, y0c[:,99:], y1c[:,99:], t0v[:,99:], t1v[:,99:], focal, mask=mask_b[:,99:])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item(); n_batch += 1
            if bi % 30 == 0:
                print(f"  E{epoch} B{bi}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        wp, wp0, wp1 = evaluate(model, valid_ds, DEV)
        print(f"  E{epoch} | loss {el/max(n_batch,1):.4f} | WP {wp:.4f} (t0={wp0:.4f} t1={wp1:.4f})", flush=True)
        if wp > best_wp: best_wp = wp
    elapsed = time.time() - t_start
    print(f"  DONE in {elapsed:.0f}s | best WP={best_wp:.4f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": name, "params": n_params, "wp": best_wp, "time_s": elapsed}

# ---------- RUNNER ----------
def main():
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f"[sweep] DEV={DEV}  features=336 (raw+m100+s100)", flush=True)

    # Load seq lists
    pf_tr = pq.ParquetFile(str(TRAIN_PARQ)); all_s = []
    for rg in range(pf_tr.num_row_groups):
        all_s.extend(pf_tr.read_row_group(rg, columns=["seq_ix"]).to_pandas()["seq_ix"].unique().tolist())
    all_s = sorted(set(int(s) for s in all_s)); rng = np.random.default_rng(0); rng.shuffle(all_s)
    train_seqs = all_s[:2000]
    pf_va = pq.ParquetFile(str(VALID_PARQ)); valid_s = []
    for rg in range(pf_va.num_row_groups):
        valid_s.extend(pf_va.read_row_group(rg, columns=["seq_ix"]).to_pandas()["seq_ix"].unique().tolist())
    valid_seqs = sorted(set(int(s) for s in valid_s))[:500]
    print(f"  train={len(train_seqs)} valid={len(valid_seqs)}", flush=True)

    train_ds = SeqDataset(TRAIN_PARQ, train_seqs)
    valid_ds = SeqDataset(VALID_PARQ, valid_seqs)

    experiments = [
        # (name, ModelClass, kwargs)
        ("09_single_gru_h128",       SingleGRU,        {"in_f": 336, "gru_h": 128}),
        ("11_single_tcn48_gru",      SingleTCN_GRU,    {"in_f": 336, "tcn_ch": 48, "gru_h": 128}),
        ("06_dual_tcn48_gru",        DualTCN_GRU,      {"in_f": 336, "tcn_ch": 48, "gru_h": 128}),
        ("08_dual_tcn32_gru",        DualTCN_GRU,      {"in_f": 336, "tcn_ch": 32, "gru_h": 128}),
        ("01_parallel_tcn_gru_gate", ParallelTCN_GRU,  {"in_f": 336, "tcn_ch": 48, "gru_h": 128}),
        ("04_dual_gru_gate",         DualGRU_Gate,     {"in_f": 336, "gru_h": 64}),
        ("05_parallel_concat",       ParallelTCN_GRU_Concat, {"in_f": 336, "tcn_ch": 48, "gru_h": 128}),
        ("03_dual_tcn_dual_gru",     DualTCN_DualGRU,  {"in_f": 336, "tcn_ch": 48, "gru_h": 64}),
    ]

    results = []
    for name, cls, kw in experiments:
        try:
            r = run_experiment(name, cls, kw, train_ds, valid_ds, DEV, epochs=2)
            results.append(r)
            with open("logs/results.json", "w") as f: json.dump(results, f, indent=2)
        except Exception as e:
            print(f"\n[EXP {name}] FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

    # Summary
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}", flush=True)
    print(f"{'Experiment':<32}{'Params':>10}{'WP':>10}{'Time':>10}", flush=True)
    print("-"*62, flush=True)
    valid = [r for r in results if "wp" in r]
    valid.sort(key=lambda x: x["wp"], reverse=True)
    for r in valid:
        print(f"{r['name']:<32}{r['params']:>10,}{r['wp']:>10.4f}{r['time_s']:>10.0f}s", flush=True)

    print("\nBaseline references (1500-2000 train, 100-500 valid):", flush=True)
    print("  v24 single TCN+GRU @ 1500 = 0.5783", flush=True)
    print("  Dual TCN+GRU @ 1500       = 0.6053", flush=True)
    print("  v17N full 10K             = 0.6734", flush=True)

if __name__ == "__main__":
    main()
