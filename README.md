# Sequential Target Forecasting with Parallel GRU-TCN Encoders

## 1. Overview

This report documents a sequence modeling approach for a dataset where the prediction targets are piecewise step-function-like rather than smoothly varying. The input consists of 20,000-step sequences with 112 raw features per step, and two targets (`t0`, `t1`) are predicted jointly.

The target behavior can be described as:

$$
y(t) \approx \begin{cases}
c_1 & \text{stable plateau} \\
\Delta & \text{transition to a new level} \\
c_2 & \text{stable plateau} \\
\vdots
\end{cases}
$$

with transitions occurring over comparatively short intervals. This structure is challenging for a plain regression model, which tends to learn the average level and produce predictions that lag or smear the transitions.

The model described here combines:
- Causal rolling features (cumsum-based, mean and standard deviation over a 100-step window)
- Target-derived class labels for regime supervision
- Parallel TCN and GRU temporal encoders
- Learned gate fusion of the two encoders
- Prediction decomposed as `level + direction · magnitude`
- Multi-objective training (classification, weighted Pearson, mean squared error, gate penalty)

---

## 2. Input and Target Structure

Each sequence has 20,000 time steps. Per step:

| Group | Count | Description |
|---|---|---|
| `i0` features | 52 | Instrument-0 prices, volumes, trade prices, trade volumes |
| `i1` features | 52 | Instrument-1, same structure |
| `a` features | 8 | Shared context features |
| **Total** | **112** | per timestep |

The two targets `t0`, `t1` are continuous values typically within `[-2, 2]`, occasionally exceeding this range. The first 99 rows of each sequence are warmup and produce no loss; the model still processes them to populate its internal state.

---

## 3. Causal Rolling Features (cumsum-based)

### 3.1 Construction

For each of the 112 raw features `x_t`, two rolling statistics are computed with a causal window of `W = 100`:

$$
\mu_t = \frac{1}{W}\sum_{j=t-W+1}^{t} x_j
$$

$$
\sigma_t = \sqrt{\frac{1}{W}\sum_{j=t-W+1}^{t} x_j^2 - \mu_t^2}
$$

The window is **causal only** — position `t` uses only `x_{t-W+1}, …, x_t`. No future values are used at any time, so the features are identical during training and inference.

### 3.2 Efficient computation with cumulative sums

Direct computation is `O(W)` per step. Instead, precompute cumulative sums:

$$
C_t = \sum_{j=1}^{t} x_j \qquad S_t = \sum_{j=1}^{t} x_j^2
$$

Then for `t > W`:

$$
\mu_t = \frac{C_t - C_{t-W}}{W}
$$

$$
\sigma_t^2 = \frac{S_t - S_{t-W}}{W} - \mu_t^2
$$

Each step is now **`O(1)`** (four array reads, two subtractions, one division, one square root). For `W = 100`, this reduces compute by ~50× versus a naive loop.

For the first `W - 1` steps, the window is padded by repeating `x_1` so that every `t` has a valid statistic:

```python
x_pad = concat([repeat(x_1, W-1), x])
C = cumsum(x_pad)
S = cumsum(x_pad ** 2)
mu[t]  = (C[t+W-1] - C[t-1]) / W
var[t] = max((S[t+W-1] - S[t-1]) / W - mu[t]**2, 0)
```

### 3.3 Why window = 100

The choice of `W = 100` was determined by two factors:

**3.3.1 Alignment with regime scale.** The labeling procedure (Section 4) uses a 100-step forward window to define plateau / rise / fall. A rolling statistic with the same window sees exactly the same time scale as the labels, so supervised training can learn the relation directly.

**3.3.2 Empirical separability.** An ANOVA F-test comparing raw, rolling-mean, and rolling-std features against the three-class labels gave:

| Feature set | Average F-score |
|---|---|
| Raw | 74,000 |
| Rolling mean (100) | 134,000 |
| **Rolling std (100)** | **296,000** |

For a targeted set of features, individual F-scores on rolling std reached **935,854** (`a5` std), versus 74,000 for the raw mean of the same feature. Rolling std of the 100-window is the strongest single statistical signal in the dataset.

**3.3.3 Why both mean and std.** They measure different things:
- Rolling **mean** captures *trend* — is the level drifting up or down.
- Rolling **std** captures *volatility state* — is the market quiet or moving.

A rise in the target is often preceded by a change in rolling std (volatility expanding) rather than by a raw price change. Both statistics were retained.

### 3.4 Feature expansion

Applying raw + rolling-mean-100 + rolling-std-100 to all 112 features gives:

$$
112 \text{ raw} + 112 \text{ mean}_{100} + 112 \text{ std}_{100} = 336 \text{ features per step}
$$

The 336 features are split into three views for the model:

| View | Source | Dim |
|---|---|---|
| `x_i0` | 52 i0 × 3 variants | 156 |
| `x_i1` | 52 i1 × 3 variants | 156 |
| `x_a` | 8 a × 3 variants | 24 |

These are later concatenated to 336 for the model input.

---

## 4. Target-Derived Class Labels

### 4.1 Purpose

The prediction problem involves a target that alternates between flat regions and transitions. A class label `{plateau, rise, fall}` is derived from the target itself and used to supervise an auxiliary classification head. The classifier is not used as the final prediction — it provides a direction signal `dir = p_rise − p_fall` that feeds into the numerical prediction (Section 6).

### 4.2 Labeling procedure

For each target sequence `y_t`, compute a **forward** W-step delta with `W = 100`:

$$
\Delta_t = y_{t+W} - y_t
$$

Then assign a class label:

$$
r_t = \begin{cases}
\text{rise} & \Delta_t > \tau \\
\text{fall} & \Delta_t < -\tau \\
\text{plateau} & \text{otherwise}
\end{cases}
$$

with threshold `τ = 0.3`.

**Note on forward-looking labels.** The label at `t` uses `y_{t+W}`. This is a *supervised training label* — it appears only in the loss computation. The model's input at time `t` contains only past features (causal rolling). No future information reaches the prediction heads.

### 4.3 Window size selection

A 20-step forward window gave near-ties in class probability across all three classes and did not teach the classifier to be decisive. A 100-step window produced the following label distribution (measured on the dataset):

| Label | Fraction |
|---|---|
| Plateau | ~50% |
| Rise | ~20% |
| Fall | ~30% |

Rise is much rarer than fall in this dataset, which biases any flat classifier toward the majority classes. Cross-entropy was replaced with a focal variant (Section 7.1) to prevent the classifier from collapsing to majority-class predictions.

---

## 5. Model Architecture

### 5.1 Parallel temporal encoders

Two encoders process the input sequence independently:

**TCN branch** — 4 dilated causal convolution blocks with kernel `k = 3` and dilations `[1, 4, 16, 64]`:

$$
h_t^{TCN} = \text{Conv1d}(x_{\leq t})
$$

Receptive field: `1 + 2·(1 + 4 + 16 + 64) = 171` steps. Output dimension: 48 channels.

The dilated convolutions detect local and intermediate-range patterns (micro-price reversals, order-book imbalance changes) without the sequential dependency of a recurrent model.

**GRU branch** — single-layer GRU with hidden size 96:

$$
h_t^{GRU} = \text{GRU}(x_t, h_{t-1}^{GRU})
$$

Output dimension: 96. LayerNorm applied after the GRU output for training stability.

The GRU maintains a hidden state that can retain information over the full 20,000-step sequence, in contrast to the TCN's fixed receptive field.

### 5.2 Why both encoders

They capture different frequencies:

| Encoder | Captures | Time scale |
|---|---|---|
| TCN | Local structure, micro-patterns | ≤ 171 steps |
| GRU | Long-range historical context | up to entire sequence |

Neither replaces the other. The TCN provides a sharp local signal; the GRU provides context. A learned gate (below) combines them.

### 5.3 Learned gate fusion

The two representations have different dimensions (48 vs 96). A linear projection brings the TCN to the GRU's dimension first:

$$
\tilde{h}_t^{TCN} = W_p \, h_t^{TCN} + b_p \qquad W_p \in \mathbb{R}^{96 \times 48}
$$

Then a per-timestep scalar gate combines the two:

$$
g_t = \sigma\!\left(W_g \,[\tilde{h}_t^{TCN} ; h_t^{GRU}] + b_g\right), \quad g_t \in (0, 1)^{96}
$$

$$
h_t^{fused} = g_t \odot h_t^{GRU} + (1 - g_t) \odot \tilde{h}_t^{TCN}
$$

`σ` is sigmoid, `[;]` is concatenation, `⊙` is element-wise multiplication. The gate is learned jointly with the rest of the model; there is no fixed split between the two encoders.

**Why a gate.** The relative reliability of the TCN and GRU depends on the current regime. During quiet periods the TCN's local view is cleaner; during transitions the GRU's longer memory is more useful. A learned gate lets the model make this trade-off per timestep and per hidden channel.

### 5.4 Prediction heads

The fused representation `h_fused` is passed to three heads per target (six heads total, then upgraded to 3-way ensembles in later variants):

**Classification head** — 3-way softmax:

$$
p_t = \text{softmax}(W_{cls} \, h_t^{fused})
$$

Direction signal:

$$
d_t = p_{\text{rise},t} - p_{\text{fall},t} \in [-1, 1]
$$

**Level head** — linear, unbounded:

$$
\ell_t = W_{\text{level}} \, h_t^{fused}
$$

**Spike head** — non-negative via softplus:

$$
s_t = \log(1 + e^{W_{\text{spike}} h_t^{fused}})
$$

### 5.5 Final prediction

$$
\boxed{\hat{y}_t = \ell_t + d_t \cdot s_t}
$$

The interpretation:
- `ℓ_t` is the base level the model believes the target is at.
- `s_t` is the magnitude of the current directional move, if any.
- `d_t` is the signed direction (positive for rise, negative for fall, near zero for plateau).

When the classifier is uncertain (all three probabilities near 1/3), `d_t ≈ 0` and the prediction is `ℓ_t`. When the classifier is confident of rise (`d_t ≈ +1`), the prediction jumps by `s_t` above `ℓ_t`. The final output is continuous, not quantized into three discrete values.

---

## 6. Training Objectives

Four losses are summed. All computed per-chunk (2000 timesteps per chunk) with the first 99 warmup rows skipped.

### 6.1 Focal cross-entropy

$$
L_{CE} = -\frac{1}{N}\sum_{i=1}^{N} (1 - p_{i,y_i})^\gamma \log p_{i,y_i}
$$

with `γ = 2`. Applied to both target classifiers.

Down-weights easy examples (mostly-plateau samples where the classifier is already correct), so the rare rise class contributes meaningful gradient. Without focal loss, the classifier predicts plateau almost always.

### 6.2 Weighted Pearson

Predictions and targets are clipped to `[-2, 2]`. Weights `w_i = |y_i|`. Weighted means:

$$
\bar{y}_w = \frac{\sum_i w_i y_i}{\sum_i w_i}, \quad \bar{\hat{y}}_w = \frac{\sum_i w_i \hat{y}_i}{\sum_i w_i}
$$

Weighted Pearson:

$$
WP = \frac{\sum_i w_i (y_i - \bar{y}_w)(\hat{y}_i - \bar{\hat{y}}_w)}{\sqrt{\sum_i w_i (y_i - \bar{y}_w)^2} \sqrt{\sum_i w_i (\hat{y}_i - \bar{\hat{y}}_w)^2}}
$$

Loss:

$$
L_{WP} = -WP
$$

This is the primary evaluation metric and is optimized directly. The weighting by `|y|` means large-move rows dominate the loss — matching the evaluation criteria of the dataset.

**Note on scale invariance.** `WP(α·y, y) = WP(y, y)` for any `α > 0`. WP alone does not force the model to predict large values; it only requires that the predictions correlate with the targets when both are projected to a common scale. This is why the MSE term exists.

### 6.3 Mean squared error on scored rows

Only rows marked `is_scored = True` (13% of all rows) are included:

$$
L_{MSE} = \frac{1}{N_s}\sum_{i:\,s_i=1} (y_i - \hat{y}_i)^2
$$

The `is_scored` mask selects the rows where `|y|` exceeds a threshold. MSE on those rows is what forces the model to predict magnitudes proportional to the true magnitudes. Without MSE, the model predicts values ~0.3 even when `y = 2.0` (WP would still be ~0.67 since it's scale-invariant, but the predictions would be visually wrong and lose the correlation signal).

### 6.4 Gate penalty

For both targets, push `p_plat` away from `1/3` (where all three classes are equally likely):

$$
L_{\text{gate}} = \frac{1}{N}\sum_{i} \left[ p_{\text{plat},i}(1 - p_{\text{plat},i}) \right]
$$

The term `p(1 - p)` peaks at `p = 0.5` and is minimized at `p = 0` or `p = 1`. This encourages the classifier to be decisive rather than hedge at `p_plat ≈ 0.45`.

### 6.5 Combined loss

$$
\boxed{L = 1.0 \cdot L_{CE} + 0.3 \cdot L_{WP} + 0.5 \cdot L_{MSE} + 0.5 \cdot L_{\text{gate}}}
$$

Weights were chosen empirically. `CE` dominates because without a working classifier, `d_t ≈ 0` and the prediction collapses to `ℓ_t` (which is essentially the mean).

### 6.6 Auxiliary heads (v27 variant only, omitted in the base configuration)

Aux heads predict `|Δy|` over the next 20 and 50 steps:

$$
L_{\text{aux}} = \text{MSE}(\hat{a}_{20}, |y_{t+20} - y_t|) + \text{MSE}(\hat{a}_{50}, |y_{t+50} - y_t|)
$$

Aux heads are train-only and not exported to the deployed model. They force `h_fused` to encode "how large is the coming move" in addition to direction, but add significant training time and caused underfitting in the tested variant.

---

## 7. Training Procedure

### 7.1 Batching and chunking

Each sequence (20,000 steps) is split into 2,000-step chunks. Between chunks:
- The GRU hidden state is carried forward.
- The gradient is **detached** at the chunk boundary. Gradients do not flow across chunks, which keeps the memory footprint bounded.

Batch size 2 to 4 sequences per step. AdamW optimizer, weight decay 1e-4.

### 7.2 Initialization

| Layer | Init |
|---|---|
| Linear projections | Xavier uniform, bias = 0 |
| TCN convs | Kaiming uniform (ReLU fan-in), bias = 0 |
| GRU `weight_hh` | Orthogonal |
| GRU `weight_ih` | Xavier uniform, bias = 0 |
| Prediction heads | Kaiming uniform (ReLU), bias = 0 |

The projections before the TCN and GRU use no activation function. Keeping them linear preserves the raw magnitude relationship between input features and the encoded representation; adding a ReLU would force all embeddings to be non-negative, which prevents the model from learning "this feature reduces the signal".

Orthogonal initialization of the GRU's recurrent weights keeps gradients well-conditioned during the 2,000-step unrolled pass.

### 7.3 Learning rate

Cosine annealing from `3e-4` to `1e-5` over the training budget.

### 7.4 Optional enhancements (tested, mixed results)

**EMA of weights.** Maintain an exponential moving average of model parameters at `decay = 0.999`. Used only at evaluation, providing a smoothed version of the model. Improves stability for long training runs but is not necessary for short runs (fewer than 2,000 optimizer steps, where the EMA has not converged).

**3-ensemble heads.** Replacing each prediction head with 3 identical heads averaged at inference. Reduces variance from random seed but requires ~3× more training steps per head to converge. In a 2-epoch test with 2,000 sequences, the ensemble heads underfit versus single heads.

**Differential learning rate.** Backbone at `1e-4`, heads at `1e-3`. Heads learn faster without destabilizing the backbone. Helps when heads are underfit relative to the backbone.

---

## 8. Evaluation

### 8.1 Metric

Same weighted Pearson correlation as Section 6.2, computed on the `is_scored` subset of rows. Predictions clipped to `[-2, 2]`.

### 8.2 Validity separation

The dataset is split:
- **Train:** sequences used for base training.
- **Validation:** held-out sequences used to select the best checkpoint by weighted Pearson. The validation set is never seen by any model during training.

### 8.3 Results

Reported configuration:

| Model | Validation Weighted Pearson |
|---|---|
| Base configuration (GRU h128, TCN 64ch) | **0.683** |
| Extended configuration (GRU h96, TCN 48ch, 6 heads) | 0.656 |
| Ensemble-of-heads variant | 0.643 |
| Ridge regression baseline (linear) | 0.585 |

The base configuration was the strongest single model. Adding capacity or new components did not improve the metric in any tested configuration.

---

## 9. Behavior on Validation Data

The most consistent observation across all experiments:

**Prediction-target shape.** The model's prediction is not a step function — it has high-frequency noise around the true target's level. This is expected: the model is regressing a continuous value that correlates with a mostly-flat step-function target. When the target makes a large move, the prediction follows after a short lag. When the target is flat, the prediction oscillates slightly around the level.

**Why not predict a perfectly flat step function.** Post-processing the prediction with a median filter, moving average, or quantization to discrete ticks was tested and all reduced weighted Pearson. The high-frequency noise correlates productively with the target at short lags; smoothing it out removes useful variance.

**Autoregressive variants.** Eight different autoregressive update formulas of the form `pred = f(prev, current)` were tested. Every one lowered the metric versus the pure `pred = level + dir · spike` formulation. This is consistent with the observation above — the model's variance on the flat regions is not wasted, and blending with `prev` removes it.

---

## 10. Limitations

1. **Classification is imprecise.** The classifier's maximum probability on any single class hovers around 0.45 on scored rows. Plateau and rise/fall are genuinely ambiguous at the 100-step horizon in this dataset. The model compensates by using the classification only as a direction weight (`d_t = p_rise - p_fall`) rather than as a hard gate.

2. **Long-range dependency remains unfitted.** A residual autocorrelation test on the trained model showed 18% correlation remaining at lag 200 steps — signal that no tested architecture captured. SSM, LSTM, wider TCN, and dual-branch gated models all failed to reduce this residual.

3. **The metric is scale-invariant.** Weighted Pearson cannot distinguish "predicts 0.3 when target is 2.0" from "predicts 2.0 when target is 2.0" if the relative ordering is preserved. The MSE term is essential for forcing magnitude, but MSE and WP can pull the model in opposite directions on any specific training run.

4. **Causal-only features.** All rolling statistics use only past values. Centered rolling statistics (which would leak future information) were never used as model inputs; they were only considered for label construction, which itself uses a forward window.

5. **Warmup rows are silent.** The first 99 rows of each sequence populate the GRU's hidden state but produce no loss. This means sequence starts are learned from implicit context, not from explicit supervision.

6. **The reported 0.683 is one configuration.** It should not be attributed to any single component without controlled ablation. Removing the TCN, removing the classifier, or removing the rolling features each individually lowered the metric in the experiments that were run.

---

## 11. Summary

A sequence model was trained on 20,000-step inputs with 112 raw features per step. Causal rolling mean and standard deviation (window 100, cumsum-based) expanded the input to 336 features. A parallel TCN (4 dilated causal conv blocks, receptive field 171) and GRU (hidden 96, LayerNorm) processed the sequence. A learned sigmoid gate fused the two encoders per timestep. Six heads predicted: a 3-way regime class, a base level, and a directional magnitude for each target. Predictions were formed as `level + (p_rise − p_fall) · magnitude`. Training combined focal cross-entropy (regime classification), weighted Pearson (metric), MSE on scored rows (magnitude), and a gate-entropy penalty. The base configuration achieved 0.683 weighted Pearson on held-out validation data. Adding components beyond this configuration — autoregressive gating, ensemble heads, dual-branch gating, state-space memory — did not improve the metric and in most cases lowered it.
