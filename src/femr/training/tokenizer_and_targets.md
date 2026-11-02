# Shared tokenizer and pretraining targets

This note describes how events are tokenized and what each pretraining objective predicts, for
NTP, MOTOR, TPP, MTPP, SurvivEHR and ICareFM, as implemented after the tokenizer unification
and the removal of task caps. Line references point at the current code.
Launch scripts: `script/lib.sh` and `script/<dataset>/<cluster>/<arch>.sh`.

MIMIC-IV numbers come from the existing caches under `/user/zj2398/cache/` and from scans of
`/shared/share_mala/zilin/storage/ora/meds_v0.6_reader` (364,627 subjects).

---

## 1. The shared tokenizer

### 1.1 One tokenizer for every model

All six models use one tokenizer class, `Tokenizer` (`models/tokenizer/tokenizer.py`;
`HierarchicalTokenizer` and `CLMBRTokenizer` remain as aliases). It is trained **once per
dataset** into `<metadata>/tokenizer/`; every model's `prepare.py` loads that same directory via
`--tokenizer_path` and keeps a byte-identical copy in its own directory. Training is reproducible
for the same data and `--num_threads` (per-shard seeds, ordered merge, tie-broken sort). All models
therefore see identical input tokens. What differs between models is only the **target** side
(section 2).

The tokenizer is "hierarchical": with an ontology, an event is represented by its own code
**and all of its ontology ancestors** that are in the vocabulary.

### 1.2 Training statistics (one pass over the train+tuning split)

`map_statistics` (`tokenizer.py`) collects, per subject, and merges across shards:

| Statistic | How it is computed |
|---|---|
| `code_counts[c]` | Fraction of subjects that have `c` **or any descendant of `c`**: each subject's raw codes are expanded with `ontology.get_all_parents` and every resulting code adds `1/num_subjects` once per subject. |
| `original_codes` | Set of raw `event.code` values actually observed (no ancestor expansion). |
| `property_samples[p]["text_counts"][s]` | For every non-code column `p` (e.g. `text_value`, `unit`, `route`, `race`), the fraction of subjects with string value `s` in that column. |
| `property_samples[p]["numeric_samples"]` | A reservoir sample (10,000) of the values of column `p` that parse as floats, each subject weighted equally. |
| `age_stats` | Mean/std of age, log age, inter-event delta and log delta, used to normalise the time features. |

Columns `code`, `time`, `visit_id`, `end` are excluded. All other columns, including ID-like
columns such as `hadm_id`, `order_id`, `emar_seq`, become numeric or text properties.

### 1.3 Three kinds of vocabulary entries

| Entry type | Keyed by | Score used for selection |
|---|---|---|
| **code** | code string (raw code or ontology ancestor) | conditional entropy score (1.4) |
| **text** | (column, string), e.g. (`text_value`, `"NEG"`) | binary entropy of the fraction of subjects with that string |
| **numeric** | (column, value interval), e.g. (`numeric_value`, [1.1, 1.3)) | fixed `-1`, so always kept and ranked first |

Important properties:

- **Text and numeric entries are per column, not per code.** All labs share one set of
  `numeric_value` bins and one set of `text_value` strings. For example, creatinine 1.2 and
  potassium 1.2 fall into the same `numeric_value` bin token, and `"NEG"` is one token for every
  urine test. The code token in the same event tells them apart.
- **Numeric bins.** Column `p` gets `max(1, int(num_numeric × frac_p))` quantile bins, where
  `num_numeric = 1000` and `frac_p` is `p`'s share of all numeric values. Bin edges are quantiles
  of the reservoir sample, de-duplicated, with the outer edges replaced by ±∞
  (`tokenizer.py`).
- A text value that parses as a float is treated as numeric for its column.

MIMIC tokenizer (`mtpp_8k_T_8_V_5`): 12,604 entries = 8,390 code + 4,026 text + 188 numeric.
Text entries by column: `text_value` 3,806, `frequency` 105, `route` 39, `race` 27, `unit` 20, …;
numeric bins: `numeric_value` 124, `emar_seq` 18, `link_order_id` 16, `order_id` 16, …

### 1.4 Entropy scores and why the code score is conditional

Write `H(q) = q·ln q + (1−q)·ln(1−q)` for the **negative** binary entropy in nats. It is `≤ 0`, and
most negative (`−ln 2 ≈ −0.693`) at `q = 0.5`.

**Code entries** (`tokenizer.py`):

```
P(parent)   = min(1, min over immediate parents of code_counts[parent])
q           = min(1, code_counts[c] / P(parent))        # = P(c | parent present)
score(c)    = P(parent) · H(q)
```

The score is exactly **minus the conditional entropy of the presence indicator of `c` given its
parent's indicator**. When the parent is absent the child is absent too, so it contributes 0:

```
H(C | Parent) = P(parent)·H_b(q) + (1 − P(parent))·0 = −score(c)
```

Codes with `q = 0` or `q = 1` are skipped. The birth code is special-cased to score `H(0.5)`.

**Text entries**: `score = H(min(1, fraction of subjects with that string))`, unconditional.

**Filter and ranking** (`tokenizer.py`):

1. Keep entries with `score ≤ H(min_fraction)`, where `min_fraction = 1/1000` and
   `H(0.001) ≈ −0.0079`, i.e. entries carrying at least ~0.0079 nats of information.
2. Sort ascending by score: numeric bins (−1) first, then the most informative codes and texts.
3. Truncate to `vocab_size` (16,384).

What the filter means in practice:

- **Codes with no ontology parent** (all of MIMIC): `P(parent) = 1`, so the rule is simply
  `0.1% ≤ prevalence ≤ 99.9%` of subjects.
- **Codes with a parent**: the child needs `P(parent)·H_b(q) ≥ 0.0079`. Since `H_b ≤ ln 2`, a child
  whose rarest immediate parent covers **less than ~1.14% of subjects can never be kept**; its
  parent's token represents it. A child present in (almost) every subject who has the parent
  (`q ≈ 1`) adds nothing beyond the parent and is dropped too.

**Why condition on the parent.** In this tokenizer, an ancestor's token is already part of every
event of its descendants (1.6). A child token is only useful to the extent that it distinguishes
subjects who share the parent, which is what `H(C | Parent)` measures. The unconditional
prevalence would rank a child that duplicates its parent as highly as the parent itself and waste
vocabulary on redundant tokens. Ranking by information also decides what survives the
`vocab_size` cut.

### 1.5 `original_codes`

`original_codes` = vocabulary **code** entries that occur as a raw `event.code` in the data
(not ancestor-only codes). They are the classes for NTP and SurvivEHR. They are computed **after**
the `vocab_size` cut, so every target class also has its own input token. On MIMIC, no code has an ontology
parent, so `original_codes` equals all 8,390 code tokens.

### 1.6 Tokenizing one event: what is summed

`get_feature_codes(event)` (`tokenizer.py`) returns token ids and weights:

1. **Code and ancestors**: every code in `ontology.get_all_parents(event.code)` (the code itself
   plus all ancestors) that is in the vocabulary, each with weight `1/n`, where `n` is the number
   of such tokens.
2. **Value tokens**: for every other column `k` of the event:
   - if its value parses as a float and `k` has numeric bins → that column's bin token, weight 1;
   - else if the string is a text entry for `k` → that text token, weight 1;
   - otherwise nothing.

The model embeds the event with `nn.EmbeddingBag(mode="sum")`
(`models/architecture/embedding.py:189-202`), so the event's vector is
`Σ weight_i · E[token_i]`, followed by RMSNorm. Age and inter-event time enter separately as time
features and positional encoding.

Examples:

| Event | Tokens summed (weights) |
|---|---|
| MIMIC lab `LAB//50912//mg/dL`, `numeric_value=1.2`, `unit="mg/dL"`, `hadm_id=…` | code (1) + `numeric_value` bin containing 1.2 (1) + `unit`=`"mg/dL"` text (1) + `hadm_id` bin (1) |
| MIMIC urine lab `LAB//51486//UNK`, `text_value="NEG"` | code (1) + `text_value`=`"NEG"` (1) |
| OMOP diagnosis with 5 in-vocabulary ancestors | code + 5 ancestors (1/6 each) |

One event is always **one sequence position**. Ancestors add tokens inside the position, not
positions, so sequence length, attention cost and `tokens_per_batch` are unaffected; only the
`hierarchical_tokens` arrays grow. Events with no in-vocabulary token at all are skipped.

### 1.7 From events to training sequences (`models/processor.py`)

- Events without a time, or on/before the birth date in the `person` table, are merged into one
  initial position (`processor.py:170-185`).
- Same-day de-duplication: an event whose tokens (ancestors included) were **all** already seen
  that day is skipped as input and as label (`processor.py:210-213`).
- All pretraining tasks have `needs_exact() = False`, so each subject contributes **one window:
  its last `max_length` events**; earlier events are not used in pretraining
  (`processor.py:66-75`).
  - Transformer: `max_length = tokens_per_batch // min_subjects_per_batch = 16384 // 2 = 8192`
    (`--min_subjects_per_batch 2`). Windows are packed without padding into batches of
    ≤ 16,384 tokens.
  - Mamba: windows capped by `mamba_per_sample_max_tokens = 8192`, sorted by length and packed so
    that `num_sequences × longest ≤ 16,384`.

---

## 2. Pretraining targets

### 2.1 What each objective predicts and where labels sit

| Model | Prediction at a position | Label positions |
|---|---|---|
| NTP | softmax over target codes: the code of the **immediately next** event | every input event whose next event's code is a target class |
| SurvivEHR | competing risk over target codes plus time to the **immediately next** event; value head for the next event's value | same as NTP |
| ICareFM | values of numeric target codes | numeric target events (`icarefm_task.py:218-245`) |
| MOTOR | time to the **next occurrence of every task code**, censored at end of record (piecewise exponential, global time bins) | positions whose next event has a later timestamp (`motor.py:242`) |
| TPP | same targets as MOTOR, per-task time bins | same as MOTOR (`tpp.py:287`) |
| MTPP (pretrain `--loss_type mtpp_shared`) | time to the next occurrence of every task code, plus value bins for numeric tasks | same as MOTOR (`mtpp.py:458`) |

### 2.2 Which events count as an occurrence of a code

All models see every event as **input** (1.6). The objectives differ in which events count as
**target** occurrences:

| Event kind (MIMIC count over vocabulary codes) | MOTOR / TPP | MTPP | NTP / SurvivEHR (event head) | ICareFM / value heads |
|---|---|---|---|---|
| no value (108M) | ✓ | ✓ | ✓ | – |
| numeric only (285M) | ✗ | ✓ | ✓ | ✓ |
| text only (276M) | ✗ | ✗ | ✓ | ✗ |
| numeric + text (48M; ordinal scales such as GCS) | ✗ | ✗ | ✓ | ✗ |
| code | `motor.py:40`, `tpp.py:40` | `mtpp.py:41-56` | `processor.py:227-238` | `survivehr_task.py` `_normalize_value`; `icarefm_task.py:46` |

The rule is **per event, not per code**. A code with both value-less and valued events stays a
MOTOR/TPP task through its value-less events. Example: `LAB//51466//UNK` has 454,836 value-less,
118,180 text-only and 1 numeric event.

### 2.3 Which codes are tasks

| Model | Candidate codes | Filters | Cap |
|---|---|---|---|
| NTP | `original_codes` | tokenizer entropy filter only | none |
| SurvivEHR | event head: `original_codes`; value head: codes with ≥ 1000 raw numeric-only events | tokenizer entropy filter only | none |
| ICareFM | codes with ≥ 1000 raw numeric-only events | – | none |
| MOTOR, TPP | **all vocabulary code tokens** (ancestors included under an ontology) | `rate == 0`; `frac_events < 1/1000` | none |
| MTPP | all vocabulary code tokens | `rate == 0`; `frac_events < 1/1000`; a code with **any** numeric value needs ≥ 1000 value samples or it is dropped entirely | none (`--num_final_tasks` only if given) |

Definitions:

- `frac_events`: fraction of label positions at which the code occurs later in the record, i.e. is
  not censored.
- `rate == 0`: the code never occurs after any label position. This is always the case for static
  codes such as `MEDS_BIRTH` and `GENDER`, and for codes all of whose events are excluded by 2.2.
- MTPP's "value samples" are counted at label positions (one per position whose next occurrence
  carries a value). SurvivEHR and ICareFM count raw numeric events. The two lists therefore differ
  slightly; on MIMIC with the current tokenizer: MTPP 1,164, SurvivEHR/ICareFM 1,081, overlap 1,076.

### 2.4 How the code sets relate

**Without an ontology (MIMIC).** Vocabulary code tokens = `original_codes` (8,390), so every model
starts from the same list:

```
MOTOR = TPP  ⊂  original_codes = NTP = SurvivEHR (event head)
MTPP         ⊂  original_codes
```

- **MOTOR and TPP have identical task codes.** Same candidates, same event rule, same label
  positions, same filters; they differ only in time bins and head.
- **MTPP vs MOTOR/TPP: neither contains the other.**
  - MTPP **adds** codes whose occurrences are numeric-only. 1,238 vocabulary codes have
    numeric-only events, and 792 of those have no value-less events, so MOTOR/TPP give them rate 0.
  - MTPP **drops** codes with a few numeric values but < 1000 value samples, even when they have
    many value-less events. On MIMIC, 60 codes fail this rule; 55 of them have value-less events
    and are kept by MOTOR/TPP, e.g. `LAB//51466//UNK`, `LAB//51508//UNK`, `LAB//51003//ng/mL`.
- **All of them are below NTP.** Filters only remove codes, and static codes always fail `rate == 0`.
- **The 971 text-valued codes** (every event carries a text value) are NTP/SurvivEHR targets only.
  They account for 311M events (~43% of events on vocabulary codes): 915 ICU flowsheet items
  (safety checks, heart rhythm, GCS components, positioning, O2 device), 44 lab results/holds and
  6 outpatient vitals stored as text. A charting time stamp holds on average ~52 of them at once.
  They are deliberately **not** targets for MOTOR/TPP/MTPP.

Rough MIMIC sizes: NTP 8,390; MTPP ≈ 7.2k (estimate: 8,390 − 971 text-valued − 60 value-dropped −
≤121 rate/frequency); MOTOR = TPP: printed at prepare.

**With an ontology (e.g. CUMC OMOP).** Vocabulary code tokens = observed codes ∪ ancestor-only codes:

```
original_codes  ⊂  vocabulary code tokens  ⊇  MOTOR/TPP/MTPP candidates
```

- MOTOR/TPP/MTPP may predict **ancestor** tasks. For no-value events, an ancestor task fires when
  any descendant occurs (`motor.py:43`, `mtpp.py:59`). Numeric events are **not** expanded to
  ancestors in MTPP (`mtpp.py:45-50`).
- NTP/SurvivEHR predict raw observed codes only. A target code `c` means "this exact code", whereas
  MOTOR/MTPP's `c` means "`c` or any descendant".
- Their task counts can be larger or smaller than NTP's: ancestors add tasks, while the event rules
  remove valued codes. The counts are printed at prepare.

### 2.5 Value targets

| Model | Value target | Codes | Normalisation |
|---|---|---|---|
| MTPP | value bin (classification, `value_bins = 5`) of the next occurrence | MTPP numeric tasks (≥ 1000 value samples) | per-code quantile bins |
| SurvivEHR | regression on the next event's value | ≥ 1000 raw numeric-only events | per-code 1st–99th percentile scaling; values outside are masked |
| ICareFM | values of target codes | same rule as SurvivEHR | per-code mean/std z-score |
| MOTOR, TPP, NTP | none | – | – |

Only numeric-only events provide value targets. Numeric + text (ordinal) events never do.
