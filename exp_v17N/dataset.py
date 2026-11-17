import json
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT = Path("/mnt/data/wnn_connectome_starterpack")
TRAIN_PARQ = ROOT / "wnn_connectome_starterpack/train_labeled.parquet"
VALID_PARQ = ROOT / "datasets/valid_labeled.parquet"
STATS_PATH = ROOT / "exp_v17N/stats_112.json"

FEATURES = (
    [f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)] +
    [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)] +
    [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)] +
    [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)] +
    [f"a{i}" for i in range(8)]
)

with open(STATS_PATH) as f:
    _stats = json.load(f)
MU = np.array(_stats["mean"], dtype=np.float32)
SIG = np.array(_stats["std"], dtype=np.float32)
SIG = np.where(SIG < 1e-8, 1.0, SIG)

def get_rolling(x, window):
    T, F = x.shape
    x_f64 = x.astype(np.float64)
    x_pad = np.concatenate([np.repeat(x_f64[:1], window-1, axis=0), x_f64], axis=0)
    csum = np.cumsum(x_pad, axis=0); csum_sq = np.cumsum(x_pad * x_pad, axis=0)
    end_sum = csum[window-1:]; end_sum_sq = csum_sq[window-1:]
    start_sum = np.concatenate([np.zeros((1, F), dtype=np.float64), csum[:-window]], axis=0)
    start_sum_sq = np.concatenate([np.zeros((1, F), dtype=np.float64), csum_sq[:-window]], axis=0)
    mean = (end_sum - start_sum) / window
    sq_mean = (end_sum_sq - start_sum_sq) / window
    var = np.maximum(sq_mean - mean*mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

def build_features(X_raw):
    X = (X_raw - MU) / SIG
    i0 = X[:, :52]; i1 = X[:, 52:104]; a = X[:, 104:112]
    
    m50_all, s50_all = get_rolling(X, 50)
    m100_all, s100_all = get_rolling(X, 100)
    
    i0_m50, s50 = m50_all[:, :52], s50_all[:, :52]
    i0_m100, s100 = m100_all[:, :52], s100_all[:, :52]
    i1_m50, s50_i1 = m50_all[:, 52:104], s50_all[:, 52:104]
    i1_m100, s100_i1 = m100_all[:, 52:104], s100_all[:, 52:104]
    a_m50, a_s50 = m50_all[:, 104:112], s50_all[:, 104:112]
    a_m100, a_s100 = m100_all[:, 104:112], s100_all[:, 104:112]
    
    # 8 raw + 8 m50 + 8 m100 + 8 s50 + 8 s100 = 40 features
    a_feat = np.concatenate([a, a_m50, a_m100, a_s50, a_s100], axis=-1)
    
    # 52 raw + 52 m50 + 52 m100 + 52 s50 + 52 s100 + 40 a = 300 features per instrument
    inst0 = np.concatenate([i0, i0_m50, i0_m100, s50, s100, a_feat], axis=-1)
    inst1 = np.concatenate([i1, i1_m50, i1_m100, s50_i1, s100_i1, a_feat], axis=-1)
    
    return np.concatenate([inst0, inst1], axis=-1).astype(np.float32)

class SeqDataset(Dataset):
    def __init__(self, parquet_path, seq_ids):
        self.parquet_path = str(parquet_path)
        self.seq_ids = seq_ids
        self.pf = pq.ParquetFile(self.parquet_path)
        self.seq_to_rg = {}
        self.has_is_scored = "is_scored" in self.pf.schema.names
        for rg in range(self.pf.num_row_groups):
            df = self.pf.read_row_group(rg, columns=["seq_ix"]).to_pandas()
            for sid in df["seq_ix"].unique(): self.seq_to_rg[int(sid)] = rg

    def __len__(self): return len(self.seq_ids)

    def __getitem__(self, idx):
        sid = self.seq_ids[idx]
        rg = self.seq_to_rg[sid]
        cols = ["seq_ix", "step_in_seq", "t0", "t1", "t0_class", "t1_class"] + FEATURES
        if self.has_is_scored:
            cols.append("is_scored")
            
        df = self.pf.read_row_group(rg, columns=cols).to_pandas().sort_values("step_in_seq")
        
        X_raw = df[FEATURES].to_numpy(np.float32)
        y0 = df["t0_class"].to_numpy(np.int64); y1 = df["t1_class"].to_numpy(np.int64)
        t0 = df["t0"].to_numpy(np.float32).copy(); t1 = df["t1"].to_numpy(np.float32).copy()
        
        if self.has_is_scored:
            mask_scored = df["is_scored"].to_numpy(np.float32)
        else:
            mask_scored = np.ones(len(df), dtype=np.float32)
            
        X_final = build_features(X_raw)
        
        return (torch.from_numpy(X_final), torch.from_numpy(y0), torch.from_numpy(y1),
                torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(mask_scored))
