"""Classification accuracy test on v17N best.pt."""
import sys, torch, numpy as np, pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SeqDataset, VALID_PARQ
from model import DualGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading best.pt...", flush=True)
ckpt = torch.load("checkpoints/best.pt", map_location=DEVICE)
model = DualGRU(hidden=64).to(DEVICE)
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()
print(f"  Loaded epoch {ckpt.get('epoch', '?')}  missing={len(missing)} unexpected={len(unexpected)}", flush=True)

pf = pq.ParquetFile(str(VALID_PARQ))
valid_seqs = []
for rg in range(pf.num_row_groups):
    df = pf.read_row_group(rg, columns=["seq_ix"]).to_pandas()
    valid_seqs.extend(df["seq_ix"].unique().tolist())
valid_seqs = sorted(set(int(s) for s in valid_seqs))[:500]
print(f"  Valid seqs: {len(valid_seqs)}", flush=True)

ds = SeqDataset(VALID_PARQ, valid_seqs)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

all_p0, all_p1, all_y0c, all_y1c, all_t0, all_t1, all_mask = [], [], [], [], [], [], []
print("Running inference...", flush=True)
with torch.no_grad():
    for bi, item in enumerate(loader):
        # v17N dataset returns 6 items
        X, y0c, y1c, t0v, t1v, mask = item[:6]
        X = X.to(DEVICE)
        out = model(X, state=None, use_ar=False)
        all_p0.append(out["l0"].argmax(-1).cpu().numpy())
        all_p1.append(out["l1"].argmax(-1).cpu().numpy())
        all_y0c.append(y0c.numpy()); all_y1c.append(y1c.numpy())
        all_t0.append(t0v.numpy()); all_t1.append(t1v.numpy())
        all_mask.append(mask.numpy())
        if bi % 10 == 0:
            print(f"  batch {bi}/{len(loader)}", flush=True)

p0 = np.concatenate([a.ravel() for a in all_p0])
p1 = np.concatenate([a.ravel() for a in all_p1])
y0c = np.concatenate([a.ravel() for a in all_y0c])
y1c = np.concatenate([a.ravel() for a in all_y1c])
t0 = np.concatenate([a.ravel() for a in all_t0])
t1 = np.concatenate([a.ravel() for a in all_t1])
mask = np.concatenate([a.ravel() for a in all_mask])

def report(name, pred, target, mask_rows, use_scored=False):
    if use_scored:
        sel = mask_rows > 0.5
        pred = pred[sel]; target = target[sel]
    n = len(pred)
    if n == 0:
        print(f"{name}: empty"); return
    acc = (pred == target).mean()
    print(f"\n{'='*60}\n{name}  (n={n:,})\n{'='*60}")
    print(f"  Overall accuracy: {acc*100:.2f}%")
    cm = np.zeros((3,3), dtype=int)
    for t, p in zip(target, pred):
        cm[t, p] += 1
    names = ["plateau", "rise", "fall"]
    print(f"  {'':<10}{'→plat':>10}{'→rise':>10}{'→fall':>10}")
    for i, nr in enumerate(names):
        row = cm[i]; rp = row / max(row.sum(), 1) * 100
        print(f"  {nr:<10}{rp[0]:>9.1f}%{rp[1]:>9.1f}%{rp[2]:>9.1f}%")
    print(f"  Recall (per class):")
    for i, nr in enumerate(names):
        rec = cm[i,i] / max(cm[i].sum(), 1) * 100
        print(f"    {nr:<10} {rec:>5.1f}%  (n={cm[i].sum():,})")
    print(f"  Precision (per class):")
    for i, nr in enumerate(names):
        prec = cm[i,i] / max(cm[:,i].sum(), 1) * 100
        print(f"    {nr:<10} {prec:>5.1f}%")

print("\n" + "#"*60 + "\n# v17N CLASSIFICATION ACCURACY\n" + "#"*60)
report("t0 (all rows)", p0, y0c, mask, use_scored=False)
report("t0 (scored only)", p0, y0c, mask, use_scored=True)
report("t1 (all rows)", p1, y1c, mask, use_scored=False)
report("t1 (scored only)", p1, y1c, mask, use_scored=True)
print("\n" + "="*60)
sc = mask > 0.5
acc_all = ((p0 == y0c) & (p1 == y1c)).mean() * 100
acc_sc = ((p0[sc] == y0c[sc]) & (p1[sc] == y1c[sc])).mean() * 100
print(f"  Joint t0+t1 accuracy (all):    {acc_all:.2f}%")
print(f"  Joint t0+t1 accuracy (scored): {acc_sc:.2f}%")
