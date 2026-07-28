# ORA abnormal-event ablations — implementation plan

Goal: test whether ICareFM's regression advantage comes from **how it parameterises the value axis**
(a threshold-crossing event with a continuously varying threshold) or from **where it spends its
capacity** (numerical variables only).

Requested variants:

- **(i) `ora-abn`** — non-numerical loss unchanged; numerical loss predicts a *dynamic-threshold
  abnormal event* instead of a (time-bin, value-bin) pair.
  *Your prediction: regression improves over ORA but stays below ICareFM; classification and TTE
  comparable to ORA.*
- **(ii) `ora-abn-numonly`** — same as (i) with the non-numerical loss removed.
  *Your prediction: behaves like ICareFM.*

Stated constraints: (a) keep both the event and censoring terms; (b) predict the *first abnormal*
event rather than the first event; (c) τ must be dynamic, sampled on the fly rather than baked into
preprocessing.

---

## 1. Assessment of the plan

**The design is sound.** Two variants that differ by exactly one component give a clean two-step
ladder ORA → (i) → (ii), and stating the expected outcomes in advance makes the experiment
falsifiable. Four things to sharpen before writing code.

**1.1 Your constraint (c) is less costly than you think — no re-preprocessing is needed (for the
recommended event definition).** τ is sampled inside `MTPP_Task.cleanup()`, which runs at
*collation* time, i.e. already on the fly, exactly as `ICareFMTask.cleanup()` does. The prepared
batches already contain everything needed: per (anchor, numerical code) the time and the **raw
value** of the next occurrence (`value_sparse.time`, `value_sparse.value`), plus the per-anchor
censoring time. Only the pre-computed *value bins* become unused. **Existing `train_batches` /
`val_batches` are reusable as-is**, so these ablations cost pretraining runs only, not a re-run of
`prepare.py`. This is a large saving and it should drive the event-definition decision below.

**1.2 Your constraint (b) hides the one genuinely expensive choice.** "First abnormal event" can mean
two different things:

- **A1 — the next measurement is abnormal.** Event = the first future occurrence of code `m`
  crosses τ. If that measurement does not cross, we have *observed evidence that no crossing had
  occurred by that time*, i.e. right-censoring at the first measurement.
- **A2 — the first abnormal measurement within a horizon.** Look at every measurement of `m` in
  `(t, t+H]` and take the first that crosses τ. This is what ICareFM does with `H = 48 h`.

**These target the same estimand** — the hazard of "first abnormal measurement". A1 is simply a more
heavily censored estimator of it, and it is statistically coherent. But A1 needs no pipeline change,
while A2 requires storing *all* future occurrences per (anchor, code) and re-running `prepare.py` on
both datasets. Recommendation: **A1**, and name it accurately in the paper ("time to the first
abnormal measurement, right-censored at the next measurement").

**1.3 The censoring branch does more work than you may expect — and that is good news.** Under A1
there are three cases per (anchor, numerical code), and two of them share one code path:

| Case | Target |
|---|---|
| Next occurrence at `t*`, value crosses τ | **Event** in the time bin of `t*` |
| Next occurrence at `t*`, value does not cross τ | **Censored** at the time bin of `t*` |
| No future occurrence of the code | **Censored** at the time bin of `censor_time` (end of record) |

So constraint (a) is satisfied automatically: "observed but not abnormal" reuses the existing
censoring term with a different censoring time. The only change to `_create_time_value_bins` is
*which* time is treated as the censoring time.

**1.4 The risk to your prediction for (i).** The improvement is not guaranteed, and the reason is
worth understanding before you spend a run. For a *single* draw of τ, an abnormal-event label carries
**less** information than ORA's V-way value-bin label ("above the 73rd percentile" ≺ "in bin 4 of 5").
The threshold objective only overtakes the categorical one *in expectation over many τ draws across
training*, and only if the head can represent a continuous response to τ. That makes the head design
(§2.3) decisive: with a head that can only express a handful of thresholds, variant (i) will likely
be **worse** than ORA on regression, not better. Design for continuous τ from the start.

**1.5 Temper the prediction for (ii).** Even with the non-numerical term removed, (ii) still differs
from our ICareFM baseline on five axes: the time grid (ORA's code-specific inter-event quantile bins
vs 48 fixed hourly bins), the horizon, the event support (A1's next measurement vs all measurements
in 48 h), the treatment of censored pairs (ORA supervises them, ICareFM drops them), and the target
code set. "Similar to ICareFM" is therefore only expected if capacity allocation is the dominant
factor. Decide in advance what you conclude if (ii) lands *between* ORA and ICareFM — that would
point at one of those five axes, and the natural follow-up is to align them one at a time.

**1.6 Two cheaper controls that may be more informative per GPU-hour.** Run at least the first.

- **C1 — reweight.** Multiply ORA's existing numerical term by α ∈ {2, 5, 10}. No code beyond a
  config flag. If this closes most of the regression gap, the explanation is loss weighting, and
  variants (i)/(ii) become far less necessary.
- **C2 — resolution.** Re-prepare with `--value_bins 20` instead of 5. Directly tests "the value
  axis is resolution-limited" while holding the loss form fixed. (Costs a `prepare.py` run.)

C1 + variant (i) answers most of the question; C1 alone may answer it.

---

## 1b. Decisions locked (2026-07-27)

| Decision | Choice | Note |
|---|---|---|
| D-1 event definition | **A1** — next occurrence only | Preserves ORA's data model exactly, so (i) differs from ORA in exactly one respect. Scan-forward is rung 3 of the ladder, not part of (i)/(ii) — see §2.D-1b |
| D-4 time bins | **T-ora, T = 8** | Matches the main ORA model; also zero cost, since existing batches were prepared with `--time_bins 8`. `T = 16` would need a task refit and adds a second changed variable |
| D-5 threshold sampling | **S1**, z-units, `\|τ\| ~ U(0,4)`, `δ ~ Bern(0.5)` | Per-code mean/std joined from `icarefm_task.pkl` |
| D-8 scope | **MIMIC-IV + Transformer first** | |
| D-2 loss form | **P2′** (revised recommendation, was P1) | See §2.D-2; satisfies constraint (a) literally and keeps ORA's loss geometry |
| D-3 head | **D2-film** | See §2.D-3 |
| D-6 granularity | one `(τ, δ)` per (anchor, numerical code), resampled each collation | |
| D-7 normalisation | **N1**, per-branch mean then `L = L_non + β·L_num`, `β = 1` | |

**Open item:** align the numerical code set between the ablation and the ICareFM baseline — see
§2.D-9. These sets are currently *different*, which would otherwise confound variant (ii).

---

## 2. Decisions — reasoning

Recommended option in **bold**.

### D-1. Event definition

| | Description | Pipeline cost | Fidelity to ICareFM |
|---|---|---|---|
| **A1** | Event = the next occurrence of the code crosses τ; censor at that occurrence if it does not | **None** — reuse existing prepared batches | Same estimand, more censoring |
| A2 | Event = first crossing among all occurrences in `(t, t+H]`; censor if none observed | Re-run `prepare.py` on MIMIC-IV and CUMC; larger batch files | Matches ICareFM |

A2 also forces a horizon choice (`H = 48 h` to match ICareFM), which conflicts with ORA's
code-specific quantile time bins — see D-4. Note ORA's last numerical time-bin edge is
`float("inf")` (`mtpp.py`, `numerical_task_time_bins[-1] = inf`), so ORA's time grid already spans
the whole remaining record; a 48 h horizon would leave the upper bins receiving only censored labels.

### D-1b. Should we scan forward past a non-crossing measurement?

Your question: when the next occurrence does not cross τ, should we keep looking at later
measurements until one crosses, and only censor at end-of-record if none ever does?

**Statistically, yes — that is the more efficient estimator of the same quantity, and it is closer to
ICareFM.** A1 censors at the first measurement, so a "no crossing" label is backed by a single
observation; scanning forward backs it by all observations up to the crossing. Neither is biased;
scanning is just more informative.

**But for variant (i) we recommend not scanning**, because ORA's own numerical loss today looks only
at the *first* future occurrence of each code. Keeping A1 means variant (i) differs from ORA in
exactly one respect — the value axis parameterisation — which is what (i) is meant to measure.
Scanning changes the event support at the same time, so a difference in results could no longer be
attributed to one cause.

If you later want the scan-forward version as rung 3 of the ORA → ICareFM ladder, it can be made
exact and compact rather than storing every future measurement:

> For direction "high", the first crossing time as a function of τ is determined entirely by the
> **upper-record subsequence** of the future values — the measurements that exceed everything before
> them. Store only those (and the lower records for the "low" direction). Chain length is `O(log n)`
> in expectation rather than `O(n)`, and it can be enumerated for any suffix in chain-length time by
> precomputing a next-greater-element table with a monotonic stack. This gives exact first-crossing
> labels for arbitrary τ at a fraction of the storage of A2.

That is a clean rung-3 design, but it needs a `prepare.py` re-run and is out of scope for (i)/(ii).

### D-2. Probabilistic form of the numerical branch

ORA today: softmax over `T × V` cells, event → `log P[t*, v*]`, censored → `log(1 − Σ_{r ≤ c*} Σ_v P[r,v])`.

| | Form | "Never crosses" representable? | Comment |
|---|---|---|---|
| **P2′** | Softmax over `T+1` outputs, the extra slot being "no crossing"; ORA's event and censoring terms otherwise | Yes, via the extra slot | **Recommended.** Keeps ORA's loss geometry exactly, so (i) is a one-variable change from ORA. Satisfies constraint (a) literally |
| P1 | Discrete hazard `λ_r = sigmoid(logit_r)`, masked BCE; event → `y[r*]=1, w[1..r*]=1`; censored → `y=0, w[1..c*]=1` | Yes — all `λ_r` can go to 0 | ICareFM's form. Strictly more general than a softmax, but changes the loss geometry as well as the event definition |
| P2 | Plain softmax over `T` (ORA's form with `V` collapsed) | **No** | Misspecified — see below. Not recommended |

**Why P2 is not recommended.** ORA's softmax asserts `Σ_r p_r = 1`: the code *will* occur, in one of
the `T` bins. That is already an approximation for "first occurrence", but it becomes materially
wrong for "first *abnormal* occurrence", because the probability of ever crossing depends strongly on
τ — at `|τ| = 4σ` the event is nearly impossible. Under P2 the model cannot say so; it must place
probability 1 somewhere, and the censoring term pushes that mass into the last time bin. The final
bin then silently absorbs two different meanings — "late" and "never" — and how much of each depends
on the sampled τ. The time distribution is therefore distorted in a τ-dependent way.

**"Never crosses" versus "censored" are different things, and only one of them is a label.**
*Censored* is a property of the observation: we stopped looking, so we do not know what happened.
*Never crosses* is a property of the distribution: mass that lies beyond the horizon. You do not need
a separate label category — censoring already handles the data side correctly, and both P2′ and P1
give the model a way to *represent* `P(never) > 0`. P2 does not, which is the whole problem.

**Is P1 inconsistent with the non-numerical branch, which predicts time only?** No — a hazard
sequence is strictly more general than a softmax over time. Hazards `λ_1..λ_T` induce
`P(event in bin r) = λ_r ∏_{j<r}(1−λ_j)` together with `P(no event) = ∏_j (1−λ_j)`. A softmax over
`T` gives the first part with the second forced to zero; a softmax over `T+1` gives both. So P2′ and
P1 describe the same family, just parameterised differently. The practical difference is loss scale:
P1 emits `T` BCE terms per (anchor, code) where P2′ emits one cross-entropy term, which is why D-7
matters much more under P1.

Use the **same** form for both (i) and (ii) so that (i) → (ii) isolates exactly one thing. If (ii)
fails to approach ICareFM under P2′, switching to P1 is the natural next thing to try, since it is
ICareFM's own parameterisation.

### D-3. How τ enters the model

**Where the value axis currently lives.** The backbone gives `H ∈ R^{P × d}` (`P` prediction points,
`d = 768`). Then

```
final_layer:            Linear(d, T·D2)      →  h ∈ R^{P × T × D2}     # one D2=512 vector per time bin
non_numerical_task_layer: Linear(D2, K_non)  →  [P, T, K_non]  → softmax over T
numerical_task_layer:     Linear(D2, V·K_num)→  [P, T·V, K_num] → softmax over T·V
```

Under an abnormal-event objective the output is a distribution over **time only**, conditioned on τ,
so the `V` axis disappears from the output. The question is how the logit at `(prediction point p,
time bin t, code k)` comes to depend on the sampled `(τ, δ)`.

| | Design | New params | τ resolution |
|---|---|---|---|
| **D2-film** | Emit `F` basis coefficients per (time bin, code) and contract them with a feature map of `(τ, δ)` | **None** if `F = V = 5` → exactly parameter-matched to ORA | Continuous |
| D3-icare | ICareFM's head: `MLP([H_t ‖ τ·u_k ‖ d_δ])` | Adds a head; not parameter-matched | Continuous |
| D1-reuse | Keep the `T × V` head, define `P(cross)` by summing value bins above/below τ, τ snapped to bin edges | None | **Only `V−1 = 4` thresholds** |

**D2-film in detail.** Replace the `V` value slots with `F` *basis coefficients*, then contract:

```
coef = Linear(D2, F·K_num)(h)              →  [P, T, F, K_num]      # same tensor shape as before
φ    = [1, τ, τ², |τ|, 2δ−1]               →  [P, K_num, F]         # τ, δ sampled per (anchor, code)
logit[p,t,k] = Σ_f coef[p,t,f,k] · φ[p,k,f]  →  [P, T, K_num]
```

The logit is a **learned low-degree polynomial in τ** whose coefficients depend on the patient state
and the time bin — i.e. the model learns a continuous response curve from threshold to risk, which is
exactly what ICareFM's threshold conditioning buys. Three properties matter here: it accepts any real
τ (no snapping); it keeps the tensor dense over all `K_num` codes, so censored pairs are still
supervised and ORA's data efficiency is preserved; and at `F = V` the parameter count is unchanged,
so the ~120M budget and the comparison to ORA stay capacity-controlled. If a degree-2 response is too
rigid, widen `φ` with fixed-centre RBF features — that only increases `F`.

**D3-icare** puts the τ-dependence through a nonlinear MLP, so it is more expressive, but its input is
`[P·K_num, 2d]`; at `P ≈ 2000` and `K_num` in the hundreds to thousands that tensor is infeasible, so
only supervised rows can be materialised. ICareFM can do that because it *drops* censored queries; ORA
wants to keep them. This is the core trade-off between the two designs.

**D1-reuse** requires τ to coincide with a value-bin edge, otherwise "is this bin above τ" is
undefined for the bin containing τ. Snapping leaves only `V−1 = 4` usable thresholds — and ORA's
existing `V`-way label already determines all four of those indicators, so this variant gives the
model strictly *less* information than ORA. Expect it to lose; keep it only as a sanity control.

### D-4. Time bins for the numerical branch

| | Option | Comment |
|---|---|---|
| **T-ora** | Keep ORA's `T = 8` code-specific inter-event-time quantile bins | Minimal change; keeps (i) one variable away from ORA |
| T-icare | Switch to 48 fixed hourly bins | Aligns with ICareFM and with the 4-hour-ahead regression protocol, but requires A2 and re-preprocessing |

Recommend **T-ora** for (i) and (ii); T-icare belongs to a later rung of the ORA → ICareFM ladder.

### D-5. Threshold sampling

| | Option | Comment |
|---|---|---|
| **S1** | `δ ~ Bern(0.5)`, `\|τ\| ~ U(0, 4)` in per-code z-units, signed by δ | Matches ICareFM (Suppl. B.6) |
| S2 | Quantile-uniform: `u ~ U(0.5, 1)` for δ=1, `u ~ U(0, 0.5)` for δ=0; τ = the code's empirical quantile at `u` | Scale-free and guarantees a controlled event rate on skewed labs; needs a finer quantile grid than the 5 bins currently stored |

**S1 requires per-code mean and standard deviation, which the MTPP task does not store.** Cheapest
source: **the ICareFM task pickle already contains exactly this** —
`ICareFMTask.value_normalization = {code: (mean, std)}` for every code with ≥ 1000 numeric
observations. Load `icarefm_task.pkl`, join on code string, and attach to the MTPP task. Zero new
computation. Fall back to one `db.map` pass (the `_value_stats_map` pattern in `icarefm_task.py`)
for any code missing from that map.

*Log the realised event rate per code.* If it collapses below ~1–2%, switch to S2.

### D-6. Sampling granularity

One `(τ, δ)` per **(anchor, numerical code)**, resampled at every collation — matching
`ICareFMTask.cleanup`, which uses `per_query` granularity. Alternatives (one draw per batch, or per
code per batch) are cheaper to vectorise but reduce within-batch diversity. **Recommend per-(anchor,
code).**

### D-7. Branch normalisation — do not skip this

`MTPPSharedTaskHead` currently accumulates both branches into `total_loss` and divides by
`total_count` (the number of marked bins). Under P1 the numerical branch emits up to `T` BCE terms
per (anchor, code) instead of one CE term, so under the current scheme **the numerical branch would
outweigh the non-numerical branch by roughly a factor of T**. Variant (i) would then silently drift
towards variant (ii).

| | Option | Comment |
|---|---|---|
| **N1** | Normalise each branch to a per-observation mean separately, then add: `L = L_non + β·L_num`, with `β = 1` by default | Keeps the branch balance comparable to ORA |
| N2 | Keep a single global count normalisation | Numerical branch dominates; not recommended |

Log both branch losses every step and confirm their ratio is comparable to the ORA run before
committing to a long job.

### D-8. Scope

Recommended: **MIMIC-IV + Transformer only** for all ablations, stated as such in the paper. Add CUMC
or Mamba only if a result is surprising. Suggested order: **C1 → (i) → (ii)**, with C2 if a
`prepare.py` slot is available.

### D-9. Numerical code set — currently different between ORA and ICareFM

This is a confound in the existing comparison, not just in the ablation. The two objectives select
their numerical targets by different rules.

| | Candidate pool | Filters |
|---|---|---|
| **ORA** (`MTPP_Task.fit_pretraining_task_info`) | the first `num_tasks = 8192` `"code"`-type vocabulary entries | `frac_events ≥ 1/1000` **and** `value_sample_count ≥ value_count_threshold`, then the whole task set is capped at `num_final_tasks = 6100` |
| **ICareFM** (`ICareFMTask.fit_pretraining_task_info`) | the full flat code list `tokenizer.original_code_to_id` | `value_sample_count ≥ value_count_threshold` only |

So ICareFM has no event-frequency filter and no task cap, while ORA draws from a truncated,
frequency-filtered pool. The two numerical target sets are therefore **not the same set**, and they
may differ substantially in size.

**Action, cheap and worth doing regardless of the ablation:** print both sets for MIMIC-IV, report
`|ORA numerical|`, `|ICareFM targets|`, and the size of their intersection. Then either

- (preferred) restrict the ablation's numerical branch to the **intersection**, so variants (i)/(ii)
  and the ICareFM baseline supervise exactly the same variables; or
- report the two set sizes in the appendix and note the difference as a caveat.

If the sets turn out to be very different, this alone could explain part of the regression gap, and
it should be checked before interpreting (i)/(ii).

---

## 3. Implementation, assuming A1 / P2′ / D2-film / T-ora (T=8) / S1 / N1

Recommended path only. Every step is additive; nothing in the existing ORA path changes.

### 3.1 New files

```
src/femr/ablation/
  plans.md                     # this file
  abn_task.py                  # ORAAbnormalTask(MTPP_Task) — τ sampling + label construction
  abn_head.py                  # MTPPAbnormalTaskHead — FiLM-conditioned hazard head
```

### 3.2 `abn_task.py` — subclass `MTPP_Task`, override `cleanup()` only

Nothing in `fit_pretraining_task_info`, `add_event`, or `get_batch_data` changes, so **prepared
batches and `mtpp_task.pkl` are reused unchanged**.

```python
class ORAAbnormalTask(MTPP_Task):
    # value_mean/value_std: np.ndarray [K_num], aligned to numerical_task_to_index_map
    # attach once after unpickling (see §3.4)
    tau_max = 4.0
    threshold_seed = None          # set on a deep copy for deterministic eval collation

    def cleanup(self, batch):
        result = super().cleanup(batch)          # non-numerical outputs kept verbatim

        # dense [P, K_num]; NaN value ⇔ no future occurrence of that code
        time_num, value_num = <the value_sparse branch of super().cleanup>
        has_event = time_num != 0

        rng = np.random if self.threshold_seed is None else np.random.default_rng(self.threshold_seed)
        P, K = time_num.shape
        direction = (rng.random((P, K)) < 0.5)                    # True = "high" event
        tau_mag   = rng.random((P, K)) * self.tau_max
        tau       = np.where(direction, tau_mag, -tau_mag)        # z-units

        z = (value_num - self.value_mean[None, :]) / self.value_std[None, :]
        crosses = np.where(direction, z >= tau, z < tau) & has_event

        # event time  = next-occurrence time when it crosses
        # censor time = next-occurrence time when it does not cross,
        #               else the per-anchor end-of-record censor time
        censor_t = np.where(has_event & ~crosses, time_num, batch["censor_time"][:, None])

        result.update({
            "abn_time_event_bin":  bin_by_task(time_num, self.numerical_time_bins, mask=crosses),
            "abn_time_censor_bin": bin_by_task(censor_t, self.numerical_time_bins, mask=~crosses),
            "abn_tau":       torch.tensor(tau,  dtype=torch.float32),
            "abn_direction": torch.tensor(direction.astype(np.int64)),
        })
        result.pop("numerical_value_event_bin", None)   # no longer used
        return result
```

`bin_by_task` is the existing per-task bucketing already inside `_create_time_value_bins`
(`time_bin_starts <= t < time_bin_ends`, with `numerical_time_bins` of shape `[K_num, T+1]`) —
factor it out rather than reimplementing.

Edge cases to handle explicitly:
- a code with no future occurrence → `has_event = False` → censored at end of record (unchanged from
  ORA);
- `value_std` missing for a code (not in the ICareFM map) → exclude that code from the numerical
  branch and log it once;
- `censor_time` earlier than the last time bin edge → already handled by the existing binning.

### 3.3 `abn_head.py` — `MTPPAbnormalTaskHead`

Copy `MTPPSharedTaskHead`; keep the non-numerical branch **verbatim** (it must be bit-identical, so
that the ORA → (i) comparison is clean). Replace the numerical branch:

```python
Fb = 5                                                 # = num_value_bins ⇒ parameter-matched
self.numerical_task_layer = nn.Linear(final_layer_size, Fb * K_num)  # same shape as before
self.never_layer          = nn.Linear(final_layer_size, Fb * K_num)  # P2' "no crossing" slot

coef  = self.numerical_task_layer(self.norm(time_independent_features)) \
            .reshape(P, self.num_time_bins, Fb, K_num)
phi   = torch.stack([torch.ones_like(tau), tau, tau**2, tau.abs(),
                     (2*direction.float() - 1)], dim=-1)             # [P, K_num, Fb]
logit = torch.einsum('ptfk,pkf->ptk', coef, phi)                     # [P, T, K_num]

# P2': one extra, time-independent "no crossing" logit per (anchor, code)
never = torch.einsum('pfk,pkf->pk',
                     self.never_layer(self.norm(time_independent_features.mean(1)))
                         .reshape(P, Fb, K_num), phi)                # [P, K_num]
p = torch.softmax(torch.cat([logit, never.unsqueeze(1)], 1), dim=1)  # [P, T+1, K_num]

# ORA's own event / censoring terms, unchanged in form:
#   event  in bin r*: log p[r*]
#   censor in bin c*: log(1 - sum_{r <= c*} p[r])   <- the never slot is inside this survival mass
L_num = -(log_p_event[event_mask].sum() + log_surv[censor_mask].sum()) \
        / (event_mask.sum() + censor_mask.sum()).clamp(min=1)

loss = L_non + self.beta * L_num                                     # N1
```

*Note on the never slot:* it must not depend on the time bin, hence the pooled
`time_independent_features.mean(1)`. `never_layer` adds `Fb·K_num·D2` parameters — small relative to
120M, but report it. To keep the count exactly matched to ORA, take the extra logit from one of the
existing `Fb` rows instead of adding a layer.

For **variant (ii)** set `self.beta_non = 0` and skip constructing `non_numerical_task_layer`. Report
the resulting parameter count; if it drops by more than ~2%, widen `final_layer_size` to rematch
~120M so the comparison stays capacity-controlled.

### 3.4 Wiring

- `prepare.py` — **no change needed** if reusing existing MTPP batches. Add
  `"ora_abn" / "ora_abn_numonly"` to `--loss_type` only so `pretrain.py` can select them.
- `pretrain.py` —
  - load `mtpp_task.pkl`, wrap in `ORAAbnormalTask`, attach `value_mean` / `value_std` from
    `icarefm_task.pkl` (join on code string; log the number of codes matched);
  - deep-copy the task with `threshold_seed = 42` for the eval collator, exactly as ICareFM does, so
    validation loss is comparable across epochs;
  - add `--abn_tau_max` (default 4.0) and `--abn_beta` (default 1.0);
  - log `L_non`, `L_num`, the realised event rate, and the fraction of censored pairs.

### 3.5 Tests before launching (`tests/test_ora_abn.py`)

1. next occurrence crosses at bin 3 → event in bin 3; loss uses `log p[3]`.
2. next occurrence does not cross, at bin 3 → censored at bin 3; loss uses `log(1 − Σ_{r≤3} p[r])`.
3. no future occurrence → censored at the end-of-record bin.
3b. `p` sums to 1 over the `T+1` slots for every (anchor, code); at `|τ| = 4` the never slot carries
   most of the mass after training on synthetic data where crossings are rare.
4. calling `cleanup` twice on the same batch changes τ and δ but not tensor shapes; with
   `threshold_seed` set it is deterministic.
5. the non-numerical outputs of `ORAAbnormalTask.cleanup` are **element-wise identical** to
   `MTPP_Task.cleanup` on the same batch (regression guard).
6. one forward/backward gives a finite loss and non-zero gradients; parameter count is within 1% of
   the ORA model.
7. sanity on the realised event rate: with `|τ| ~ U(0,4)` on standardised values, the fraction of
   crossing labels should land roughly in 5–25%. If it is below 1%, switch to S2 before training.

---

## 4. Reading the results

| Outcome | Conclusion |
|---|---|
| (i) closes most of the ORA → ICareFM regression gap while classification and TTE hold | Best case. The **value parameterisation** is the cause, and this stops being an ablation and becomes a candidate improvement to ORA itself |
| (i) does not close it but (ii) does | **Capacity allocation** is the cause. The trade-off is intrinsic, and the paper's framing is confirmed as written |
| **(i) is worse than ORA and (ii) ≈ ICareFM** | Two readings, and you must separate them before concluding (see the diagnostic below). If it is not an artefact, the reading is: the abnormal-event parameterisation does not help on its own, and ICareFM's regression advantage is *bought* by discarding non-numerical supervision. Furthermore, (i) < ORA means the non-numerical term and the threshold term actively interfere. **This is a good result for the paper** — it is the cleanest possible statement of the trade-off ORA is designed to avoid, and it says the advantage is not something ORA could pick up for free |
| **Both (i) and (ii) worse than ICareFM** | Neither the value parameterisation nor the capacity allocation explains the gap, so the cause is one of the remaining axes in §1.5. The most plausible is **protocol alignment**: ICareFM's 48 hourly bins sit exactly at the scale of the 4-hour-ahead regression task, while ORA's time bins are code-specific inter-event quantiles with an infinite last edge. Two follow-ups: check D-9 (are the numerical code sets even the same?), then run rung 2 of the ladder (48 hourly bins) to test it directly. This outcome should be reported as a limitation of the regression evaluation — it would mean the ORA/ICareFM regression comparison is not purely about the objective |
| C1 (reweighting) alone closes it | The gap was a loss-weighting artefact. Report the tuned weight and revisit the main table |

**Diagnostic to run whenever (i) underperforms ORA.** Distinguish "the threshold objective is genuinely
worse" from "the head never learned to use τ":

1. Measure the τ-sensitivity of the trained head: for held-out anchors, sweep τ over `[-4, 4]` and
   record the spread of the predicted crossing probability. A near-flat response means the
   conditioning failed and the result is an artefact, not a finding.
2. Check the realised crossing rate logged during training (§3.5, test 7). Below ~1% means the
   labels were almost all negative and S1 should be replaced by S2.
3. Confirm `L_non` is numerically unchanged from the ORA run — if it drifted, D-7 normalisation is
   the culprit and (i) has silently become (ii).

**Report classification and TTE for every variant.** ORA's claim is balance; a variant that buys
regression by discarding code-level structure has reproduced ICareFM's failure mode and should be
reported as such.

## 5. Optional follow-ups, in priority order

1. **C1 reweighting** (α ∈ {2, 5, 10}) — cheapest test of the whole hypothesis.
2. **ICareFM + non-numerical presence events** — the mirror experiment (ICareFM's own "Mixed"
   objective, Suppl. B.6). If adding code events to ICareFM recovers classification but costs
   regression, while ORA gets both, that is the strongest possible defence of the paper.
3. **The ORA → ICareFM ladder** — apply, in order: A1 → A2 (all measurements in a horizon), T-ora →
   T-icare (48 hourly bins), supervise → drop censored pairs, ORA's numerical codes → ICareFM's
   target set. Each rung named, each rung reporting all three task families.
