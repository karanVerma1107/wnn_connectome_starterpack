import time, random, sys, argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SeqDataset, TRAIN_PARQ, VALID_PARQ
from model import DualGRU, FocalLoss, total_loss

EVAL_CHUNK = 2000

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

@torch.no_grad()
def eval_valid(model, valid_ds, device, use_ar):
    model.eval()
    loader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=0)
    P0, T0, P1, T1, M, PLAT0 = [], [], [], [], [], []
    for X, _, _, t0v, t1v, mask in loader:
        X = X.to(device)
        B, T, _ = X.shape
        p0_all = torch.zeros(B, T, device=device); p1_all = torch.zeros(B, T, device=device)
        state = None
        for c0 in range(0, T, EVAL_CHUNK):
            c1 = min(c0 + EVAL_CHUNK, T)
            if state is not None: state = tuple(t.detach() for t in state)
            out = model(X[:, c0:c1, :], state=state, use_ar=use_ar)
            p0_all[:, c0:c1] = out["pred0"]; p1_all[:, c0:c1] = out["pred1"]
            state = out["state"]
        P0.append(p0_all.cpu().numpy().ravel()); P1.append(p1_all.cpu().numpy().ravel())
        T0.append(t0v.numpy().ravel()); T1.append(t1v.numpy().ravel())
        M.append(mask.numpy().ravel()); PLAT0.append(out["p_plat0"].mean().item())
    p0 = np.concatenate(P0); t0 = np.concatenate(T0)
    p1 = np.concatenate(P1); t1 = np.concatenate(T1)
    m = np.concatenate(M)
    wp = 0.5 * (compute_wp(p0, t0, m) + compute_wp(p1, t1, m))
    return wp, np.mean(PLAT0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--start-train-idx", type=int, default=0)
    ap.add_argument("--max-train-seqs", type=int, default=6500)
    ap.add_argument("--n-valid", type=int, default=1873)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--use-ar", action="store_true") # NEW
    args = ap.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR = Path("exp_v17N/checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  H: {args.hidden}  Chunk: {args.chunk}  AR: {args.use_ar}  Train: {args.max_train_seqs}  Resume: {args.resume}  Freeze: {args.freeze_backbone}", flush=True)

    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    pf_tr = pq.ParquetFile(str(TRAIN_PARQ))
    all_seqs = []
    for rg in range(pf_tr.num_row_groups):
        all_seqs.extend(pf_tr.read_row_group(rg, columns=["seq_ix"]).to_pandas()["seq_ix"].unique().tolist())
    all_seqs = sorted(set(int(s) for s in all_seqs))
    rng = np.random.default_rng(0); rng.shuffle(all_seqs)
    end_idx = args.start_train_idx + args.max_train_seqs
    train_seqs = all_seqs[args.start_train_idx : end_idx]

    pf_va = pq.ParquetFile(str(VALID_PARQ))
    valid_seqs = []
    for rg in range(pf_va.num_row_groups):
        valid_seqs.extend(pf_va.read_row_group(rg, columns=["seq_ix"]).to_pandas()["seq_ix"].unique().tolist())
    valid_seqs = sorted(set(int(s) for s in valid_seqs))[:args.n_valid]

    train_ds = SeqDataset(TRAIN_PARQ, train_seqs)
    valid_ds = SeqDataset(VALID_PARQ, valid_seqs)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = DualGRU(hidden=args.hidden).to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    if args.resume:
        resume_path = Path("exp_v17N/checkpoints/best.pt")
        if resume_path.exists():
            print(f"Resuming from {resume_path}", flush=True)
            ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        print("Training FROM SCRATCH (random init)", flush=True)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if any(key in name for key in ['gru_i0', 'gru_i1', 'gate', 'P1']):
                param.requires_grad = False
        print("Backbone frozen.", flush=True)

    lr = 2e-4 if args.resume else 3e-4
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    focal = FocalLoss(gamma=2.0)
    best_wp = -1.0; no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start = time.time(); epoch_loss = 0.0
        for bi, (X, y0c, y1c, t0v, t1v, _) in enumerate(train_loader):
            X, y0c, y1c, t0v, t1v = [t.to(DEVICE) for t in (X, y0c, y1c, t0v, t1v)]
            opt.zero_grad(set_to_none=True)
            state = None; batch_loss = 0.0
            for c0 in range(0, X.shape[1], args.chunk):
                c1 = min(c0 + args.chunk, X.shape[1])
                if state is not None: state = tuple(t.detach() for t in state)
                out = model(X[:, c0:c1, :], state=state, use_ar=args.use_ar) # Pass use_ar
                state = out["state"]
                y0_c, y1_c = y0c[:, c0:c1], y1c[:, c0:c1]
                v0_c, v1_c = t0v[:, c0:c1], t1v[:, c0:c1]
                if c0 == 0:
                    out_s = {k: (out[k][:, 99:] if out[k].dim() == 3 else out[k][:, 99:]) for k in ["l0", "l1", "pred0", "pred1", "p_plat0", "p_plat1"]}
                    y0_c, y1_c = y0_c[:, 99:], y1_c[:, 99:]
                    v0_c, v1_c = v0_c[:, 99:], v1_c[:, 99:]
                else:
                    out_s = out
                loss = total_loss(out_s, y0_c, y1_c, v0_c, v1_c, focal)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                batch_loss += loss.item()
            epoch_loss += batch_loss
            if bi % 50 == 0:
                print(f"  E {epoch} | B {bi}/{len(train_loader)} | loss={batch_loss:.4f} | {time.time()-t_start:.0f}s", flush=True)

        torch.save({"epoch": epoch, "model_state": model.state_dict()}, CKPT_DIR / f"epoch_{epoch:02d}.pt")
        print("  Running full validation eval...", flush=True)
        t_eval = time.time()
        wp, plat_mean = eval_valid(model, valid_ds, DEVICE, args.use_ar)
        print(f"=== Epoch {epoch} | loss {epoch_loss/len(train_loader):.4f} | Valid WP: {wp:.4f} | Plat Mean: {plat_mean:.3f} | Eval {time.time()-t_eval:.0f}s ===", flush=True)

        if wp > best_wp + 1e-4:
            best_wp = wp; no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, CKPT_DIR / "best.pt")
            print(f"  -> new best", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("early stop", flush=True)
                break

if __name__ == "__main__":
    main()
