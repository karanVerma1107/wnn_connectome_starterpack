"""Parallel TCN+GRU ONNX streaming benchmark. TCN and GRU both process 336 features in parallel."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, onnx
import onnxruntime as ort
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ParallelTCN_GRU, decode_heads, FocalLoss, total_loss

TOTAL_ROWS = 37_460_000
IN_F = 336
TCN_CH = 48
GRU_H = 128


class StreamingParallel(nn.Module):
    """Streaming export for ParallelTCN_GRU:
    TCN buffers (4) + GRU hidden (1) = 5 state tensors."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Precompute TCN conv weights as 2D matmuls
        tcn = model.tcn
        for i, block in enumerate(tcn.convs):
            conv, res = block.conv, block.res
            k, ic, oc = tcn.k, conv.in_channels, conv.out_channels
            W = conv.weight.detach().permute(0, 2, 1).reshape(oc, k * ic).contiguous()
            B = conv.bias.detach().clone() if conv.bias is not None else None
            setattr(self, f"cW_{i}", W)
            if B is not None: setattr(self, f"cB_{i}", B)
            if isinstance(res, nn.Conv1d):
                rW = res.weight.detach().reshape(oc, ic).contiguous()
                rB = res.bias.detach().clone() if res.bias is not None else None
                setattr(self, f"rW_{i}", rW)
                if rB is not None: setattr(self, f"rB_{i}", rB)

    def _conv_step(self, x_cur, buf, i):
        tcn = self.model.tcn
        d, k = tcn.dils[i], tcn.k
        new_buf = torch.cat([buf[:, 1:], x_cur.unsqueeze(1)], dim=1)
        ctx = torch.stack([new_buf[:, -1 - (k-1-j)*d] for j in range(k)], dim=1).reshape(1, -1)
        h = F.linear(ctx, getattr(self, f"cW_{i}"), getattr(self, f"cB_{i}"))
        return F.relu(h), new_buf

    def forward(self, x, b0, b1, b2, b3, gru_h):
        # --- TCN branch ---
        h, nb0 = self._conv_step(x, b0, 0)
        res = F.linear(x, self.rW_0, self.rB_0) if hasattr(self, "rW_0") else x
        xc = h + res
        h, nb1 = self._conv_step(xc, b1, 1)
        res = F.linear(xc, self.rW_1, self.rB_1) if hasattr(self, "rW_1") else xc
        xc = h + res
        h, nb2 = self._conv_step(xc, b2, 2)
        res = F.linear(xc, self.rW_2, self.rB_2) if hasattr(self, "rW_2") else xc
        xc = h + res
        h, nb3 = self._conv_step(xc, b3, 3)
        res = F.linear(xc, self.rW_3, self.rB_3) if hasattr(self, "rW_3") else xc
        tcn_out = h + res

        # Project TCN to gru_h
        t = self.model.proj_tcn(tcn_out)             # (1, 128)

        # --- GRU branch (directly on 336 features) ---
        gru_out, gru_h_new = self.model.gru(x.unsqueeze(1), gru_h)  # (1,1,128)
        g_out = self.model.proj_gru(gru_out.squeeze(1))             # (1, 128)

        # --- Gate fusion ---
        gate = self.model.gate(torch.cat([t, g_out], dim=-1))       # (1, 128)
        fused = t * gate + g_out * (1 - gate)                       # (1, 128)

        # --- Heads ---
        out = decode_heads(self.model.heads, fused)
        pred = torch.stack([out["pred0"], out["pred1"]], dim=-1)

        return (pred, nb0, nb1, nb2, nb3, gru_h_new)


def get_shapes(model):
    shapes = []
    for blk, d in zip(model.tcn.convs, model.tcn.dils):
        shapes.append((1, (model.tcn.k-1)*d + 1, blk.conv.in_channels))
    return shapes


def fresh_state(model, device="cpu"):
    sh = get_shapes(model)
    return {
        "b0": torch.zeros(*sh[0], device=device),
        "b1": torch.zeros(*sh[1], device=device),
        "b2": torch.zeros(*sh[2], device=device),
        "b3": torch.zeros(*sh[3], device=device),
        "gru_h": torch.zeros(1, 1, GRU_H, device=device),
    }


if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    print(f"[bench] IN_F={IN_F}  TCN_CH={TCN_CH}  GRU_H={GRU_H}", flush=True)

    model = ParallelTCN_GRU(in_f=IN_F, tcn_ch=TCN_CH, gru_h=GRU_H).to(DEVICE).eval()
    print(f"[bench] params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"[bench] RF: {model.receptive_field}", flush=True)

    wrapper = StreamingParallel(model).eval()
    shapes = get_shapes(model)
    print(f"[bench] buffer shapes: {shapes}", flush=True)

    # Correctness: streaming vs batched
    print("\n[1] Correctness", flush=True)
    torch.manual_seed(0)
    T = 300
    x_batch = torch.randn(1, T, IN_F)
    with torch.no_grad():
        out_batch = model(x_batch)
    bp0 = out_batch["pred0"][0].numpy()
    bp1 = out_batch["pred1"][0].numpy()

    state = fresh_state(model)
    sp0, sp1 = [], []
    for t in range(T):
        with torch.no_grad():
            pred, nb0, nb1, nb2, nb3, nh = wrapper(
                x_batch[0, t].unsqueeze(0),
                state["b0"], state["b1"], state["b2"], state["b3"], state["gru_h"])
        state.update({"b0": nb0, "b1": nb1, "b2": nb2, "b3": nb3, "gru_h": nh})
        sp0.append(pred[0, 0].item()); sp1.append(pred[0, 1].item())

    d0 = np.abs(bp0 - np.array(sp0)).max()
    d1 = np.abs(bp1 - np.array(sp1)).max()
    print(f"  Full (t=0..{T}): |Δp0|={d0:.2e}  |Δp1|={d1:.2e}", flush=True)
    print(f"  {'✅ PASS' if d0 < 5e-3 and d1 < 5e-3 else '❌ FAIL'}", flush=True)

    # ONNX export
    print("\n[2] ONNX export", flush=True)
    onnx_path = Path("parallel.onnx")
    state = fresh_state(model)
    dummy_x = torch.randn(1, IN_F)
    torch.onnx.export(
        wrapper,
        (dummy_x, state["b0"], state["b1"], state["b2"], state["b3"], state["gru_h"]),
        str(onnx_path),
        input_names=["x", "b0", "b1", "b2", "b3", "gru_h"],
        output_names=["pred", "nb0", "nb1", "nb2", "nb3", "nh"],
        opset_version=17, dynamic_axes=None, do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    n_nodes = len(onnx.load(str(onnx_path)).graph.node)
    print(f"  ONNX: {size_mb:.2f} MB, {n_nodes} nodes", flush=True)

    # Speed
    print("\n[3] Speed benchmark", flush=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    x_arr = np.random.randn(1, IN_F).astype(np.float32)
    state_np = {k: v.numpy().astype(np.float32) for k, v in fresh_state(model).items()}
    KEYS = ["b0", "b1", "b2", "b3", "gru_h"]
    def run_once(st):
        feed = {"x": x_arr, **{k: st[k] for k in KEYS}}
        out = sess.run(None, feed)
        for k, v in zip(KEYS, out[1:]): st[k] = v

    for _ in range(500): run_once(state_np)
    N = 5000
    state_np = {k: v.copy() for k, v in state_np.items()}
    t0 = time.perf_counter()
    for _ in range(N): run_once(state_np)
    elapsed = time.perf_counter() - t0

    per_row_us = elapsed / N * 1e6
    total_min = TOTAL_ROWS / (N / elapsed) / 60

    print(f"  {N} rows in {elapsed:.2f}s", flush=True)
    print(f"  Per row:  {per_row_us:.1f} μs", flush=True)
    print(f"  Full test: {total_min:.1f} min", flush=True)
    print(f"  {'✅ PASS' if total_min < 60 else '❌ FAIL'}", flush=True)

    print("\n" + "="*60)
    print("SUMMARY: Parallel TCN + GRU (48ch, h128, 336 feat)")
    print("="*60)
    print(f"  Nodes:     {n_nodes}")
    print(f"  Size:      {size_mb:.2f} MB")
    print(f"  Per row:   {per_row_us:.1f} μs")
    print(f"  Full test: {total_min:.1f} min")
    print(f"  Verdict:   {'✅ VIABLE' if total_min < 60 else '❌ TOO SLOW'}")
