"""Standalone training for ParallelTCN_GRU. Reuses exp_sweep/runner.py for model+dataset."""
import sys, time, json, argparse, random
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import (
    ParallelTCN_GRU, SeqDataset, TRAIN_PARQ, VALID_PARQ,
    FocalLoss, total_loss,
)

def compute_wp(pred, y, mask):
    v = mask > 0.5
    p = np.clip(pred[v], -2, 2); yc = np.clip(y[v], -2, 2)
    w = np.abs(yc); W = w.sum()
    if W < 1e-12: return 0.0
    pb = (w*p).sum()/W; yb = (w*yc).sum()/W
    cov = (w*(p-pb)*(yc-yb)).sum()/W
    vp = (w*(p-pb)**2).sum()/W; vy = (w*(yc-yb)**2).sum()/W
    return float(cov/np.sqrt(vp*vy+1e-12))

def evaluate(model, ds, DEV):
    model.eval(); P0,T0,P1,T1,M = [],[],[],[],[]
    with torch.no_grad():
        for X, _, _, t0v, t1v, mask in DataLoader(ds, batch_size=2, shuffle=False):
            X = X.to(DEV)
            out = model(X, state=None, use_ar=False)
            P0.append(out["pred0"].cpu().numpy().ravel())
            P1.append(out["pred1"].cpu().numpy().ravel())
            T0.append(t0v.numpy().ravel()); T1.append(t1v.numpy().ravel())
            M.append(mask.numpy().ravel())
    p0,t0,p1,t1,m = [np.concatenate(x) for x in (P0,T0,P1,T1,M)]
    wp0 = compute_wp(p0,t0,m); wp1 = compute_wp(p1,t1,m)
    return 0.5*(wp0+wp1), wp0, wp1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn-ch", type=int, default=48)
    ap.add_argument("--gru-h", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-train-seqs", type=int, default=7500)
    ap.add_argument("--n-valid", type=int, default=1873)
    args = ap.parse_args()

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT = Path("checkpoints"); CKPT.mkdir(exist_ok=True)
    print(f"[parallel] tcn_ch={args.tcn_ch} gru_h={args.gru_h} "
          f"train={args.max_train_seqs} valid={args.n_valid} epochs={args.epochs}", flush=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    pf_tr = pq.ParquetFile(str(TRAIN_PARQ)); all_s = []
    for rg in range(pf_tr.num_row_groups):
        all_s.extend(pf_tr.read_row_group(rg, columns=["seq_ix"]).to_pandas()["seq_ix"].unique().tolist())
    all_s = sorted(set(int(s) for s in all_s)); rng = np.random.default_rng(0); rng.shuffle(all_s)
    train_seqs = all_s[:args.max_train_seqs]

    pf_va = pq.ParquetFile(str(VALID_PARQ)); valid_s = []
    for rg in range(pf_va.num_row_groups):
        valid_s.extend(pf_va.read_row_group(rg, columns=["seq_ix"]).to_pandas()["seq_ix"].unique().tolist())
    valid_seqs = sorted(set(int(s) for s in valid_s))[:args.n_valid]
    print(f"  train_seqs={len(train_seqs)} valid_seqs={len(valid_seqs)}", flush=True)

    train_ds = SeqDataset(TRAIN_PARQ, train_seqs)
    valid_ds = SeqDataset(VALID_PARQ, valid_seqs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ParallelTCN_GRU(in_f=336, tcn_ch=args.tcn_ch, gru_h=args.gru_h).to(DEV)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}  RF={model.receptive_field}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    focal = FocalLoss()

    best_wp = -1.0; no_improve = 0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train(); t = time.time(); el = 0.0
        for bi, (X, y0c, y1c, t0v, t1v, mask_b) in enumerate(train_loader):
            X, y0c, y1c, t0v, t1v, mask_b = [z.to(DEV) for z in (X, y0c, y1c, t0v, t1v, mask_b)]
            opt.zero_grad(set_to_none=True)
            out = model(X, state=None, use_ar=False)
            out_s = {k: out[k][:, 99:] for k in ["l0","l1","pred0","pred1","p_plat0","p_plat1"]}
            loss = total_loss(out_s, y0c[:, 99:], y1c[:, 99:], t0v[:, 99:], t1v[:, 99:],
                              focal, mask=mask_b[:, 99:])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item()
            if bi % 50 == 0:
                print(f"  E{epoch} B{bi}/{len(train_loader)} loss={loss.item():.4f} {time.time()-t:.0f}s", flush=True)
        sched.step()

        torch.save({"epoch": epoch, "model_state": model.state_dict()}, CKPT / f"epoch_{epoch:02d}.pt")

        print(f"  Eval E{epoch}...", flush=True)
        t_eval = time.time()
        wp, wp0, wp1 = evaluate(model, valid_ds, DEV)
        print(f"=== [parallel] E{epoch} | loss {el/len(train_loader):.4f} | "
              f"WP {wp:.4f} (t0={wp0:.4f} t1={wp1:.4f}) | {time.time()-t_eval:.0f}s ===", flush=True)
        history.append({"epoch": epoch, "loss": el/len(train_loader), "wp": wp, "wp0": wp0, "wp1": wp1})
        with open(CKPT / "history.json", "w") as f: json.dump(history, f, indent=2)

        if wp > best_wp + 1e-4:
            best_wp = wp; no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, CKPT / "best.pt")
            print(f"  -> new best ({wp:.4f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True); break

    print(f"\n[parallel] Best WP: {best_wp:.4f}", flush=True)

if __name__ == "__main__":
    main()
