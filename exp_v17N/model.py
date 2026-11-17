import torch
import torch.nn as nn
import torch.nn.functional as F

def wp_loss(pred, target):
    y = target.clamp(-2, 2); w = y.abs(); p = pred.clamp(-2, 2)
    W = w.sum() + 1e-6
    p_mean = (w * p).sum() / W; y_mean = (w * y).sum() / W
    cov = (w * (p - p_mean) * (y - y_mean)).sum() / W
    var_p = (w * (p - p_mean)**2).sum() / W
    var_y = (w * (y - y_mean)**2).sum() / W
    return -cov / torch.sqrt(var_p * var_y + 1e-6)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-1)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

class DualGRU(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gru_i0 = nn.GRU(300, hidden, batch_first=True)
        self.gru_i1 = nn.GRU(300, hidden, batch_first=True)
        
        self.gate = nn.Sequential(nn.Linear(hidden*2, hidden), nn.Sigmoid())
        self.P1 = nn.Linear(hidden, hidden, bias=False)
        
        self.head_t0_cls = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 3))
        self.head_t1_cls = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 3))
        self.head_t0_level = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 1))
        self.head_t1_level = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 1))
        self.head_t0_spike = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 1))
        self.head_t1_spike = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 1))

        self.w_plat = nn.Parameter(torch.tensor(2.0))
        self.w_rise = nn.Parameter(torch.tensor(-2.0))
        self.w_fall = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, state=None, use_ar=False):
        B, T, _ = x.shape
        if state is None:
            h0 = torch.zeros(1, B, self.hidden, device=x.device)
            h1 = torch.zeros(1, B, self.hidden, device=x.device)
            prev0 = torch.zeros(B, device=x.device); prev1 = torch.zeros(B, device=x.device)
        else:
            h0, h1, prev0, prev1 = state

        x0 = x[:, :, :300]
        x1 = x[:, :, 300:]
        
        out0, h0_new = self.gru_i0(x0, h0)
        out1, h1_new = self.gru_i1(x1, h1)
        
        g = self.gate(torch.cat([out0, out1], dim=-1))
        h_fused = out0 + g * self.P1(out1)
        
        l0 = self.head_t0_cls(h_fused); l1 = self.head_t1_cls(h_fused)
        p0 = torch.softmax(l0, dim=-1); p1 = torch.softmax(l1, dim=-1)
        
        plat0, rise0, fall0 = p0[...,0], p0[...,1], p0[...,2]
        plat1, rise1, fall1 = p1[...,0], p1[...,1], p1[...,2]
        dir0 = rise0 - fall0; dir1 = rise1 - fall1
        
        level0 = self.head_t0_level(h_fused).squeeze(-1); level1 = self.head_t1_level(h_fused).squeeze(-1)
        spike0 = F.softplus(self.head_t0_spike(h_fused).squeeze(-1)); spike1 = F.softplus(self.head_t1_spike(h_fused).squeeze(-1))
        feat0 = level0 + dir0 * spike0; feat1 = level1 + dir1 * spike1
        
        if not use_ar:
            # PHASE 1: Direct absolute prediction (No Python loop, pure vectorized)
            pred0 = feat0
            pred1 = feat1
            # For state continuity, just pass the last absolute value as prev
            prev0 = feat0[:, -1].detach(); prev1 = feat1[:, -1].detach()
        else:
            # PHASE 2: AR Loop with fast parallel scan (blocks of 50)
            winner0 = p0.argmax(-1); winner1 = p1.argmax(-1)
            w_p = torch.sigmoid(self.w_plat); w_r = torch.sigmoid(self.w_rise); w_f = torch.sigmoid(self.w_fall)
            
            gate0 = torch.where(winner0 == 0, w_p, torch.where(winner0 == 1, w_r, w_f))
            gate1 = torch.where(winner1 == 0, w_p, torch.where(winner1 == 1, w_r, w_f))
            
            BLOCK = 50
            preds0, preds1 = [], []
            for i in range(0, T, BLOCK):
                b0 = gate0[:, i:i+BLOCK]; f0 = feat0[:, i:i+BLOCK]
                b1 = gate1[:, i:i+BLOCK]; f1 = feat1[:, i:i+BLOCK]
                
                # Parallel scan: y_t = a_t * y_{t-1} + (1-a_t)*f_t
                # A_cum = cumprod(a); B_cum = cumsum((1-a)*f / A_cum)
                A0 = torch.cumprod(b0, dim=1)
                B0 = torch.cumsum((1 - b0) * f0 / (A0 + 1e-8), dim=1)
                block_pred0 = A0 * (prev0.unsqueeze(1) + B0)
                
                A1 = torch.cumprod(b1, dim=1)
                B1 = torch.cumsum((1 - b1) * f1 / (A1 + 1e-8), dim=1)
                block_pred1 = A1 * (prev1.unsqueeze(1) + B1)
                
                preds0.append(block_pred0); preds1.append(block_pred1)
                prev0 = block_pred0[:, -1]; prev1 = block_pred1[:, -1]
                
            pred0 = torch.cat(preds0, dim=1); pred1 = torch.cat(preds1, dim=1)
        
        return {"l0": l0, "l1": l1, "pred0": pred0, "pred1": pred1, "p_plat0": plat0, "p_plat1": plat1, "state": (h0_new, h1_new, prev0, prev1)}

def total_loss(out, y0_cls, y1_cls, y0_val, y1_val, focal):
    ce = focal(out["l0"].reshape(-1, 3), y0_cls.reshape(-1)) + focal(out["l1"].reshape(-1, 3), y1_cls.reshape(-1))
    wp = wp_loss(out["pred0"], y0_val) + wp_loss(out["pred1"], y1_val)
    gate_pen = (out["p_plat0"] * (1 - out["p_plat0"])).mean() + (out["p_plat1"] * (1 - out["p_plat1"])).mean()
    return ce + 0.3 * wp + 0.5 * gate_pen
