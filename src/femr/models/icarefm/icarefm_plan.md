# ICareFM-Style Dynamic-Threshold Time-to-Event Objective

## 0. Scope

Implement an **ICareFM-style dynamic-threshold time-to-event (TTE) pretraining objective** on top of an existing EHR tokenizer and backbone.

Assumptions:

- Tokenization is fixed. In our setting, use the same tokenizer as other baselines, e.g. `clmbr_tokenizer`.
- Backbone architecture is fixed.
- This implementation only adds:
  1. target numerical-code selection,
  2. prediction-anchor generation,
  3. dynamic threshold and direction sampling,
  4. first-event survival labels,
  5. valid-loss masks,
  6. threshold/direction-conditioned survival heads,
  7. masked discrete-time survival loss,
  8. inference from hazards to cumulative risk.

This objective mimics the ICareFM dynamic-threshold survival pretraining idea while keeping our existing tokenization and architecture.

---

## 1. Core Idea

For a patient history up to prediction time `t`, a target numerical code `k`, a threshold `tau`, and a direction `delta`, predict the probability that the value of code `k` will **first cross the threshold** within a future horizon.

The model outputs a discrete hazard vector:

```python
logits[t, k, :]  # shape [R]
hazard[t, k, r] = sigmoid(logits[t, k, r])
```

where:

```python
R = T_surv = 48
r = 1, 2, ..., R
```

Interpretation:

```text
hazard[t,k,r] =
P(first threshold-crossing event for code k occurs in future bin r
  | no crossing in previous bins 1...(r-1),
    history up to t,
    threshold tau,
    direction delta)
```

This is a **discrete-time survival objective**, not a MOTOR-style piecewise-exponential continuous-time likelihood.

---

## 2. Prediction Anchor Times

Because we keep the existing tokenizer/backbone, prediction times should be constructed from the patient timeline rather than from a dense hourly grid.

For each patient `i`:

1. Sort all events by timestamp.
2. Let `unique_times_i` be all unique timestamps in the patient timeline.
3. Use a timestamp `t` as a prediction anchor only if there exists at least one **future measurement** of a selected target numerical code after `t`.

Recommended anchor definition:

```python
anchor_times_i = [
    t for t in unique_times_i
    if exists event e such that:
        e.time > t
        and e.code in target_codes
        and e has observed numerical value
]
```

This avoids anchors where no selected numerical target can ever provide future supervision.

For each anchor `t`, obtain the backbone representation:

```python
H_t = representation of patient history up to time t
```

Implementation options:

- If the backbone returns token-level hidden states, use the hidden state of the last token with timestamp `<= t`.
- If multiple events share timestamp `t`, use the hidden state after consuming all events at timestamp `t`.
- Do not use future events after `t` in `H_t`.

---

## 3. Target Numerical Code Selection

Let `target_codes` be the selected numerical codes used by the survival objective.

In our paper, use a SurvivEHR-like frequency filter:

```python
target_codes = [
    code for code, stats in value_stats_map.items()
    if stats["count"] > value_count_threshold
]
```

Requirements:

- Each target code must have observed numerical values.
- The selected code set defines the target dimension `K`.
- Treatments or non-numerical events should not be included unless explicitly intended.

Store code-specific scaling statistics:

```python
value_stats_map[code] = {
    "count": int,
    "mean": float,
    "std": float,
    "log_transform": bool,  # optional
}
```

---

## 4. Data Needed for Label Construction

For each patient, keep observed numerical measurements for selected target codes:

```python
target_events = [
    {
        "time": timestamp,
        "code": target_code,
        "value_raw": raw_numeric_value,
        "value_scaled": scaled_numeric_value,
    },
    ...
]
```

Scale continuous values as:

```python
value_scaled = (value_raw - mean_k) / std_k
```

If a log transform is used, apply it before standardization:

```python
value_scaled = (log(value_raw + eps) - mean_k) / std_k
```

Important:

- Use observed values for label construction.
- Never construct labels from imputed values.
- If multiple measurements for the same code occur at the same timestamp, aggregate them first, e.g. median, or keep the last one consistently.

---

## 5. Time Discretization for Survival Bins

The survival horizon is divided into `R=48` future bins.

If using hourly bins, define bin `r` as:

```text
(t + (r-1) hours, t + r hours]
```

Then:

```python
r = ceil((event_time - t) / 1 hour)
```

Keep an event in the survival horizon if:

```python
1 <= r <= R
```

If timestamps are irregular, still map future events to hourly-offset bins using elapsed time in hours.

---

## 6. Dynamic Threshold and Direction Sampling

For each training anchor `(patient i, time t)` and target code `k`, sample an event definition.

### 6.1 Direction

Sample:

```python
delta = 1  # high event: value >= tau
delta = 0  # low event:  value < tau
```

### 6.2 Threshold magnitude

Sample threshold magnitude in standardized units:

```python
tau_mag ~ Uniform(0, tau_max)
tau_max = 4.0
```

Because values are standardized, `tau_mag=2.0` roughly means two standard deviations from the code-specific mean.

### 6.3 Signed threshold

Use a signed threshold:

```python
if delta == 1:
    tau = +tau_mag
else:
    tau = -tau_mag
```

This gives:

```python
if delta == 1:
    event = value_scaled >= tau
else:
    event = value_scaled < tau
```

This avoids degenerate low-event definitions such as `value < +3`, which would be almost always true.

---

## 7. Event Annotation

For a fixed anchor `(i,t)`, target code `k`, threshold `tau`, and direction `delta`, inspect only future observed measurements of code `k`.

For every future observed measurement `e` of code `k`:

```python
dt_hours = (e.time - t) in hours
r = ceil(dt_hours)
```

If `1 <= r <= R`, define whether the measurement crosses the sampled threshold:

```python
if delta == 1:
    is_event = e.value_scaled >= tau
else:
    is_event = e.value_scaled < tau
```

Only observed measurements can define events or non-events.

---

## 8. First-Event Survival Label

For each anchor `(i,t,k,tau,delta)`, create:

```python
y = zeros([R])
w = zeros([R])
```

Let `future_measurements` be all observed measurements of target code `k` in `(t, t+R hours]`.

### 8.1 Current already abnormal

Before constructing future labels, check whether code `k` is observed at the current anchor time `t`.

If an observed measurement of code `k` at `t` already crosses the sampled threshold:

```python
current_event = True
```

then:

```python
w[:] = 0
```

Reason: the time-to-first-event is `0`, but the model predicts only `r=1..R`. This anchor should not contribute loss for this `(t,k,tau,delta)` query.

### 8.2 First future event exists

If the current state is not already event-positive and the first future threshold-crossing measurement occurs in bin `r_star`:

```python
y[r_star - 1] = 1
w[0 : r_star] = 1
w[r_star : R] = 0
```

Example: first event at `t+10h`

```text
r:       1 2 3 4 5 6 7 8 9 10 11 ... 48
y:       0 0 0 0 0 0 0 0 0  1  0 ...  0
w:       1 1 1 1 1 1 1 1 1  1  0 ...  0
```

Offsets after the first event are excluded because this is a **time-to-first-event** objective.

### 8.3 No event but future measurement exists

If there is no threshold-crossing event in `(t, t+R]`, but there is at least one observed future measurement of code `k` in the horizon:

```python
y[:] = 0
w[:] = 1
```

This provides negative survival evidence:

```text
no threshold-crossing event occurred within R bins
```

### 8.4 No future measurement

If there is no observed future measurement of code `k` in `(t, t+R]`:

```python
w[:] = 0
```

Reason: absence of measurement is not evidence of normality.

---

## 9. Summary of Weight Logic

For each anchor `(i,t,k,tau,delta)`:

| Case | Label/weight behavior |
|---|---|
| Current value at `t` already crosses threshold | `w[:] = 0` |
| First future event at bin `r_star` | `y[r_star-1]=1`, `w[0:r_star]=1`, `w[r_star:R]=0` |
| No future event, but at least one future observed measurement | `y[:]=0`, `w[:]=1` |
| No future observed measurement | `w[:]=0` |

This implementation uses a **future-evidence-only** censoring rule:

```python
not_censored = any observed measurement of code k in (t, t+R]
```

This differs slightly from a paper-style `t:t+R` window, but is clearer for tokenized patient timelines and avoids treating the current measurement alone as evidence for the future.

---

## 10. Model Head

Given backbone representation:

```python
H_t  # shape [D]
```

For each target code `k`, create a threshold- and direction-conditioned survival head.

### 10.1 Code-specific head

For each target code `k`:

```python
u_k: learnable vector, shape [D_cond // 2]
direction_embedding: nn.Embedding(2, D_cond // 2)
MLP_k: maps [D + D_cond] -> [R]
```

Forward pass:

```python
tau_embed = tau * u_k                    # [D_cond // 2]
dir_embed = direction_embedding(delta)   # [D_cond // 2]
cond = concat(tau_embed, dir_embed)      # [D_cond]

head_input = concat(H_t, cond)           # [D + D_cond]
logits = MLP_k(head_input)               # [R]
hazard = sigmoid(logits)                 # [R]
```

### 10.2 Shared-head alternative

If one MLP per target code is too expensive:

```python
code_embed = nn.Embedding(K, D_code)
threshold_embed = tau * threshold_code_embed[k]
dir_embed = direction_embedding(delta)

head_input = concat(H_t, code_embed[k], threshold_embed, dir_embed)
logits = shared_mlp(head_input)
```

The shared-head version is easier to implement but is not exactly the original ICareFM design.

---

## 11. Loss Function

For each anchor, compute binary cross entropy over the `R` discrete hazards:

```python
bce = F.binary_cross_entropy_with_logits(
    logits,       # [R]
    y.float(),    # [R]
    reduction="none",
)

loss_anchor = (w.float() * bce).sum() / (w.float().sum() + 1e-8)
```

For a batch of anchors:

```python
loss = (w * bce).sum() / (w.sum() + 1e-8)
```

Recommended: normalize per target code to prevent dense codes from dominating:

```python
loss_per_code = []
for k in target_codes:
    loss_k = (w_k * bce_k).sum() / (w_k.sum() + 1e-8)
    loss_per_code.append(loss_k)

loss = mean(loss_per_code)
```

---

## 12. Efficient Training Data Generation

Avoid constructing a dense `[B, T, K, R]` tensor if the patient timeline is long.

Use sampled anchor tuples:

```python
anchor = {
    "patient_id": i,
    "time": t,
    "target_code": k,
    "tau": tau,
    "delta": delta,
    "hidden_state": H_t,
    "y": y,  # [R]
    "w": w,  # [R]
}
```

Recommended sampling strategy:

1. Sample patients.
2. For each patient, compute valid `anchor_times_i`.
3. Sample a subset of anchor times `t`.
4. For each `t`, sample a subset of target codes `k` that have future measurements after `t`.
5. For each `(t,k)`, sample `tau` and `delta`.
6. Build `y` and `w`.
7. Keep only anchors with `w.sum() > 0`.

---



## 14. Inference

At inference, choose:

```python
target code k
raw clinical threshold tau_raw
direction delta
horizon h <= R
prediction time t
```

### 14.1 Convert raw threshold to scaled threshold

```python
if log_transform[k]:
    tau_raw = log(tau_raw + eps)

tau_scaled = (tau_raw - mean_k) / std_k
```

For high event:

```python
delta = 1
tau = tau_scaled
```

For low event:

```python
delta = 0
tau = tau_scaled
```

### 14.2 Predict hazards and cumulative risk

```python
logits = survival_head(H_t, k, tau, delta)  # [R]
hazard = sigmoid(logits)                    # [R]

survival = cumprod(1 - hazard, dim=-1)
risk = 1 - survival
risk_h = risk[h - 1]
```

Mathematically:

```text
F_k(h | H_t, tau, delta)
= 1 - product_{r=1}^h (1 - lambda_k(r | H_t, tau, delta))
```

---

## 15. Edge Cases

### 15.1 Anchor too close to sequence end

If `t + R` exceeds the end of the patient timeline, it is still valid if there are observed future measurements before the end of the record.

However, for simplicity, either:

- keep anchors and use observed measurements within available follow-up, or
- drop anchors with insufficient follow-up if you want a fixed full horizon.

Recommended for simplest implementation:

```python
drop anchors if no future target-code measurement exists after t
```

### 15.2 Current state already abnormal

If code `k` is observed at `t` and already crosses the sampled threshold:

```python
w[:] = 0
```

Reason: the event time is 0, outside the predicted support `r=1..R`.

### 15.3 No future measurement

If there is no observed future measurement of code `k` after `t` within the horizon:

```python
w[:] = 0
```

Do not treat absence of measurement as a negative label.

### 15.4 Multiple future abnormal measurements

If abnormal measurements occur at both `t+10h` and `t+15h`, only the first one is used:

```python
y[9] = 1
w[0:10] = 1
w[10:R] = 0
```

This is time-to-first-event survival, not repeated-event modeling.

---

## 16. Unit Tests

### Test 1: First future event

Input:

```text
event at offsets 10 and 15
```

Expected:

```text
y[9] = 1
y[14] = 0
w[0:10] = 1
w[10:R] = 0
```

### Test 2: Fully negative observed future horizon

Input:

```text
no event in horizon
at least one future observed measurement
```

Expected:

```text
all y = 0
all w = 1
```

### Test 3: No future measurement

Input:

```text
no observed measurement of code k in (t, t+R]
```

Expected:

```text
all w = 0
```

### Test 4: Current already abnormal

Input:

```text
observed value at t crosses threshold
```

Expected:

```text
all w = 0
```

### Test 5: Cumulative risk

Input:

```text
hazard = [0.1, 0.2, 0.3]
```

Expected:

```text
survival = (1-0.1)(1-0.2)(1-0.3) = 0.504
risk = 1 - 0.504 = 0.496
```

---

## 17. Suggested File Structure

```text
icarefm_objective/
  target_codes.py
    select_target_codes()
    compute_value_stats()

  anchors.py
    get_unique_patient_times()
    get_valid_anchor_times()
    sample_anchor_times()
    get_hidden_state_at_time()

  dynamic_tau.py
    sample_tau_delta()
    raw_to_scaled_threshold()
    is_threshold_event()

  labels.py
    build_first_event_label()
    build_survival_weight()
    build_anchor_target()

  heads.py
    ThresholdConditionedSurvivalHead
    MultiCodeSurvivalHeads
    SharedSurvivalHead

  loss.py
    masked_bce_survival_loss()
    masked_bce_survival_loss_per_code()

  inference.py
    hazards_to_cumulative_risk()
    predict_threshold_crossing_risk()

  tests/
    test_anchor_generation.py
    test_dynamic_tau.py
    test_label_generation.py
    test_loss.py
    test_inference.py
```

---

## 18. Difference from MOTOR

ICareFM-style dynamic-threshold objective:

```text
lambda_r is a probability in [0,1]
loss is masked BCE over discrete hazards
time bins are future hourly offsets
risk = 1 - product(1 - lambda_r)
```

MOTOR-style objective:

```text
lambda_r is a continuous-time rate
loss includes delta * log(lambda) - lambda * U
time pieces may have arbitrary interval lengths
```

For this baseline, implement the ICareFM-style discrete hazard objective.

---

## 19. Final Checklist

- [ ] Target codes are selected numerical codes with enough observed values.
- [ ] Prediction anchors are unique patient timestamps with future target-code measurements.
- [ ] Backbone representation `H_t` only uses history up to `t`.
- [ ] Values used for label construction are observed, not imputed.
- [ ] Values and thresholds are in the same standardized scale.
- [ ] Dynamic tau samples both direction and threshold magnitude.
- [ ] `delta=1` means high event: `value >= tau`.
- [ ] `delta=0` means low event: `value < tau`.
- [ ] If current value already crosses threshold, set `w[:] = 0`.
- [ ] If first future event occurs, include loss before and including the first event.
- [ ] If no future event but at least one future measurement exists, include all-negative survival loss.
- [ ] If no future measurement exists, set `w[:] = 0`.
- [ ] Loss uses `binary_cross_entropy_with_logits`.
- [ ] Inference converts hazards to cumulative risk using `1 - cumprod(1 - hazard)`.
