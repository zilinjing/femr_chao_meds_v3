from __future__ import annotations

import datetime
import functools
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import meds_reader
import numpy as np
import torch

import femr.models.config
import femr.models.tokenizer
import femr.ontology
import femr.stat_utils
from femr.models.tasks.ntp import Task


def _try_float(value: Any) -> Optional[float]:
    """Return a finite float value, or None for missing/non-numeric values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _value_stats_map(
    subjects: Iterable[meds_reader.Subject], *, original_codes: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Map step for fitting ICareFM numerical targets and global value normalization.

    The target-selection rule intentionally follows the plan: a value counts only
    when it is stored as a numeric MEDS value and not as text. This avoids treating
    categorical/text measurements as continuous ICareFM targets.
    """
    original_code_set = set(original_codes)
    stats: Dict[str, Dict[str, Any]] = {}

    for subject in subjects:
        for event in subject.events:
            if event.code not in original_code_set:
                continue
            if event.numeric_value is None or event.text_value is not None:
                continue

            value = _try_float(event.numeric_value)
            if value is None:
                continue

            if event.code not in stats:
                stats[event.code] = {
                    "count": 0,
                    "value_stats": femr.stat_utils.OnlineStatistics(),
                }
            stats[event.code]["count"] += 1
            stats[event.code]["value_stats"].add(1, value)

    return stats


def _value_stats_agg(first: Dict[str, Dict[str, Any]], second: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Reduce step that merges per-worker value counts and online statistics."""
    for code, second_stats in second.items():
        if code not in first:
            first[code] = {
                "count": 0,
                "value_stats": femr.stat_utils.OnlineStatistics(),
            }

        first[code]["count"] += second_stats["count"]
        first[code]["value_stats"].combine(second_stats["value_stats"])

    return first


def build_first_event_label(
    future_bins: Sequence[int],
    future_values: Sequence[float],
    tau: float,
    direction: int,
    t_surv: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one ICareFM first-threshold-crossing label.

    A query is `(anchor time, target code, threshold tau, direction)`. The label
    is a discrete survival target over hourly bins:

    - first future crossing occurs at bin r: y_r is one and bins through r are
      observed;
    - future measurements exist but none cross: all bins are observed negatives;
    - no future measurement exists: all weights stay zero.
    """
    y = np.zeros(t_surv, dtype=np.float32)
    w = np.zeros(t_surv, dtype=np.float32)

    if len(future_values) == 0:
        return y, w

    first_event_bin = None
    for bin_idx, value in zip(future_bins, future_values):
        # Measurements outside the configured horizon cannot supervise this query.
        if bin_idx < 1 or bin_idx > t_surv:
            continue
        is_event = value >= tau if direction == 1 else value < tau
        if is_event:
            first_event_bin = int(bin_idx)
            break

    if first_event_bin is not None:
        y[first_event_bin - 1] = 1
        w[:first_event_bin] = 1
    else:
        w[:] = 1

    return y, w


class ICareFMTask(Task):
    """ICareFM-style dynamic-threshold task on FEMR/MTPP prediction anchors.

    This task deliberately does not create ICareFM's original dense hourly grid.
    It lets `processor.py` choose token-timestamp anchors through the same
    `needs_exact=False` path used by MTPP, then stores only the observed numerical
    measurements needed to construct threshold queries later during collation.
    """

    @classmethod
    def fit_pretraining_task_info(
        cls,
        db: meds_reader.SubjectDatabase,
        tokenizer: femr.models.tokenizer.CLMBRTokenizer,
        value_count_threshold: int = 1000,
        t_surv: int = 48,
        tau_max: float = 4.0,
    ) -> "ICareFMTask":
        """Fit target-code selection and global numeric normalization on training data."""
        # Preserve tokenizer order so target indices are stable across fitting,
        # serialization, and model-head construction.
        original_codes = [
            code for code, _ in sorted(tokenizer.original_code_to_id.items(), key=lambda item: item[1])
        ]

        # `db.map` shards subjects across workers. Each worker returns counts and
        # online mean/std state, then `_value_stats_agg` combines them exactly.
        value_stats = functools.reduce(
            _value_stats_agg,
            db.map(functools.partial(_value_stats_map, original_codes=original_codes)),
            {},
        )

        target_codes: List[str] = []
        value_normalization: Dict[str, Tuple[float, float]] = {}
        for code in original_codes:
            # The support threshold is the only target-code filter requested.
            stats = value_stats.get(code)
            if stats is None or stats["count"] < value_count_threshold:
                continue

            code_stats = stats["value_stats"]
            mean = float(code_stats.mean())
            std = float(code_stats.standard_deviation()) if code_stats.count > 1 else 1.0
            # Constant-valued targets still need a finite denominator. Keeping
            # std=1 makes the scaled value well-defined without inventing spread.
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0

            target_codes.append(code)
            value_normalization[code] = (mean, std)

        if len(target_codes) == 0:
            raise ValueError("ICareFMTask.fit_pretraining_task_info found no eligible numerical target codes.")

        print(f"ICareFM task fitted with {len(target_codes)} numerical target codes")

        return cls(
            target_codes=target_codes,
            value_normalization=value_normalization,
            t_surv=t_surv,
            tau_max=tau_max,
        )

    def __init__(
        self,
        target_codes: Sequence[str],
        value_normalization: Mapping[str, Tuple[float, float]],
        t_surv: int = 48,
        tau_max: float = 4.0,
        threshold_seed: Optional[int] = None,
    ):
        # These fields are persisted in the task pickle and become the contract
        # between prepared sparse evidence and the ICareFM head.
        self.target_codes = list(target_codes)
        self.target_code_set = set(self.target_codes)
        self.code_to_target_idx = {code: i for i, code in enumerate(self.target_codes)}
        self.value_normalization = {
            code: (float(mean), float(std)) for code, (mean, std) in value_normalization.items()
        }
        self.t_surv = int(t_surv)
        self.tau_max = float(tau_max)
        self.threshold_seed = threshold_seed

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        # Only model-head parameters belong in the task config. Value
        # normalization stays in the task object because it is used during batch
        # preparation, not by the neural head.
        return femr.models.config.FEMRTaskConfig(
            task_type="icarefm",
            task_kwargs=dict(
                target_codes=self.target_codes,
                t_surv=self.t_surv,
                tau_max=self.tau_max,
            ),
        )

    def start_subject(self, subject: meds_reader.Subject, _ontology: Optional[femr.ontology.Ontology]) -> None:
        """Collect selected numerical measurements for one subject.

        `add_event` is called repeatedly after this. Keeping a sorted per-subject
        measurement list lets each anchor gather future evidence without looking
        back into raw MEDS events.
        """
        self._subject_measurements: List[Tuple[datetime.datetime, int, float]] = []
        self._future_cursor = 0

        for event in subject.events:
            if event.time is None or event.code not in self.target_code_set:
                continue
            if event.numeric_value is None or event.text_value is not None:
                continue

            value = _try_float(event.numeric_value)
            if value is None:
                continue

            mean, std = self.value_normalization[event.code]
            scaled_value = (value - mean) / std
            target_idx = self.code_to_target_idx[event.code]
            # time, idx and value
            measurement = (event.time, target_idx, float(scaled_value))
            self._subject_measurements.append(measurement)

        # Future lookup assumes chronological order. Duplicate timestamps are
        # allowed; `add_event` advances the cursor past same-time measurements.
        self._subject_measurements.sort(key=lambda item: item[0])

        # Per-subject sparse arrays mirror CSR layout. They are indexed by the
        # subject's candidate anchors before `processor.py` subsamples/slices them.
        self.per_subject_future: Dict[str, List[Any]] = {
            "target_idx": [],
            "bin_idx": [],
            "value_scaled": [],
            "indptr": [0],
        }

    def needs_exact(self) -> bool:
        # Match MTPP's subsampled-anchor path in `map_preliminary_batch_stats`.
        return False

    def get_sampled_labels(self, length: int) -> int:
        # Use the MTPP default so sequence windows sample the same number of
        # prediction anchors for fair comparison.
        return max(5, length // 10)

    def start_batch(self) -> None:
        """Initialize batch-level sparse arrays for the selected anchors."""
        self.batch_future: Dict[str, List[Any]] = {
            "target_idx": [],
            "bin_idx": [],
            "value_scaled": [],
            "indptr": [0],
        }

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        """Copy only the selected per-subject anchor rows into the batch arrays."""
        for index in subject_label_offsets:
            # Copy future measurements in this anchor row.
            future_start = int(self.per_subject_future["indptr"][index])
            future_end = int(self.per_subject_future["indptr"][index + 1])
            self.batch_future["target_idx"].extend(self.per_subject_future["target_idx"][future_start:future_end])
            self.batch_future["bin_idx"].extend(self.per_subject_future["bin_idx"][future_start:future_end])
            self.batch_future["value_scaled"].extend(self.per_subject_future["value_scaled"][future_start:future_end])
            self.batch_future["indptr"].append(len(self.batch_future["target_idx"]))

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]] = None,
        actually_add: bool = True,
    ) -> int:
        """Register one candidate prediction anchor. We save target code, time bin and normalized value in a sparse list and keep track of offsets indptr

        The processor attaches returned labels to the previous token position.
        `current_date` is therefore the anchor time, while `next_date` is only
        used to reject repeated timestamps and the final no-future call.
        """
        if next_date is None or next_date == current_date:
            return 0

        # Advance once, chronologically, so future evidence is strictly after
        # the anchor. Same-time measurements are context/past labels, not future
        # targets, matching MTPP's cursor convention.
        while (
            self._future_cursor < len(self._subject_measurements)
            and self._subject_measurements[self._future_cursor][0] <= current_date
        ):
            self._future_cursor += 1

        if self._future_cursor >= len(self._subject_measurements):
            return 0

        # Preliminary conversion only needs to know whether this candidate anchor
        # exists; the sparse row is appended only during actual batch creation.
        if not actually_add:
            return 1

        horizon = datetime.timedelta(hours=self.t_surv)
        horizon_end = current_date + horizon
        bin_width = datetime.timedelta(hours=1)

        # Store observed future measurements within the survival horizon. Missing
        # measurements remain missing; absence is not treated as a normal value.
        for event_time, target_idx, value_scaled in self._subject_measurements[self._future_cursor:]:
            if event_time > horizon_end:
                break
            delta = event_time - current_date
            bin_idx = int(math.ceil(delta / bin_width))
            if 1 <= bin_idx <= self.t_surv:
                self.per_subject_future["target_idx"].append(target_idx)
                self.per_subject_future["bin_idx"].append(bin_idx)
                self.per_subject_future["value_scaled"].append(value_scaled)

        self.per_subject_future["indptr"].append(len(self.per_subject_future["target_idx"]))

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        """Serialize batch-level sparse evidence into numpy arrays for HF datasets."""
        return {
            "future_indptr": np.array(self.batch_future["indptr"], dtype=np.int32),
            "future_target_idx": np.array(self.batch_future["target_idx"], dtype=np.int32),
            "future_bin_idx": np.array(self.batch_future["bin_idx"], dtype=np.int16),
            "future_value_scaled": np.array(self.batch_future["value_scaled"], dtype=np.float32),
        }

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Sample thresholds and convert sparse evidence into query tensors.

        This runs inside `processor.collate`, after saved batches are loaded.
        Keeping threshold sampling here ensures train/validation batch files do
        not contain fixed thresholds or fixed crossing labels.
        """
        # HuggingFace datasets may hand back torch tensors. Convert to numpy for
        # lightweight sparse-row iteration and dynamic query construction.
        future_indptr = _to_numpy(batch["future_indptr"]).astype(np.int64)
        future_target_idx = _to_numpy(batch["future_target_idx"]).astype(np.int64) # indicating codex index of each position eg: [2, 2, 0, 0, 1] if we have three codes in total
        future_bin_idx = _to_numpy(batch["future_bin_idx"]).astype(np.int64) # indicating time index of each position eg: [1, 7, 1, 6, 8] if there are 8 bins
        future_value_scaled = _to_numpy(batch["future_value_scaled"]).astype(np.float32)

        # how many unique prediction points, like 2
        num_anchors = max(int(future_indptr.shape[0]) - 1, 0)
        # code counts of each prediction points, if future_indptr = [0,2,5], counts= [2,3]
        counts = np.diff(future_indptr)
        if num_anchors == 0 or future_target_idx.shape[0] == 0 or int(counts.sum()) == 0:
            # Return shape-stable empty tensors so the model can produce a zero
            # loss for sparse batches with no valid dynamic query.
            return self._empty_cleanup_result()

        # corresponding anchor index for each element in the indptr: [0,0,1,1,1]
        measurement_anchor_idx = np.repeat(np.arange(num_anchors, dtype=np.int64), counts)
        # normally it should be all valid
        valid = (
            (future_target_idx >= 0)
            & (future_target_idx < len(self.target_codes))
            & (future_bin_idx >= 1)
            & (future_bin_idx <= self.t_surv)
        )
        if not bool(valid.all()):
            measurement_anchor_idx = measurement_anchor_idx[valid]
            future_target_idx = future_target_idx[valid]
            future_bin_idx = future_bin_idx[valid]
            future_value_scaled = future_value_scaled[valid]
            if future_target_idx.shape[0] == 0:
                return self._empty_cleanup_result()

        # Unique (anchor, target) pairs define query rows. Targets without future observations in this horizon would be zero-weight queries, so they are
        # omitted from the materialized tensors. -> in theory we have num_anchor_points * len(self.target_codes) in total
        pair_key = measurement_anchor_idx * len(self.target_codes) + future_target_idx

        # each prediction point can have repeated measurements for one code, we only want to generate one tau and direction for each type of code
        # so we map these codes to a single (same anchor,target) and generate tau and direction
        unique_pair_key, measurement_query_idx = np.unique(pair_key, return_inverse=True)

        query_anchor_idx = unique_pair_key // len(self.target_codes)
        query_target_idx = unique_pair_key % len(self.target_codes)
        num_queries = int(unique_pair_key.shape[0])

        # Dynamic ICareFM query sampling with independent thresholds/directions
        # for every materialized (anchor, target) query.
        rng = self._cleanup_rng()
        query_direction = (rng.random(num_queries) < 0.5).astype(np.int64)
        tau_mag = (rng.random(num_queries) * self.tau_max).astype(np.float32)
        query_tau = np.where(query_direction == 1, tau_mag, -tau_mag).astype(np.float32)

        measurement_direction = query_direction[measurement_query_idx]
        measurement_tau = query_tau[measurement_query_idx]
        crosses = np.where(
            measurement_direction == 1,
            future_value_scaled >= measurement_tau,
            future_value_scaled < measurement_tau,
        )

        # first_event_bin = [49, 49, 49] fill num_queries (unique anchor-code pair) shape vector with self.t_surv + 1
        first_event_bin = np.full(num_queries, self.t_surv + 1, dtype=np.int64)
        # replace anchor-code pair in first_event_bin with time bin index ti if it cross the boundary in ti
        if bool(crosses.any()):
            np.minimum.at(first_event_bin, measurement_query_idx[crosses], future_bin_idx[crosses])

        query_y = np.zeros((num_queries, self.t_surv), dtype=np.float32)
        query_weight = np.ones((num_queries, self.t_surv), dtype=np.float32)
        # it is only true if cross happens, otherwise false
        has_event = first_event_bin <= self.t_surv
        if bool(has_event.any()):
            event_rows = np.flatnonzero(has_event)
            query_y[event_rows, first_event_bin[event_rows] - 1] = 1
            bins = np.arange(1, self.t_surv + 1, dtype=np.int64)[None, :]
            query_weight[event_rows] = (bins <= first_event_bin[event_rows, None]).astype(np.float32)
            # bins is like [[1, 2, 3, 4, 5, 6, 7, 8]], it is only true before and at the first_event_bin index

        return {
            "query_anchor_idx": torch.tensor(query_anchor_idx, dtype=torch.long),
            "query_target_idx": torch.tensor(query_target_idx, dtype=torch.long),
            "query_tau": torch.tensor(query_tau, dtype=torch.float32),
            "query_direction": torch.tensor(query_direction, dtype=torch.long),
            "query_y": torch.tensor(query_y, dtype=torch.float32),
            "query_weight": torch.tensor(query_weight, dtype=torch.float32),
        }

    def _cleanup_rng(self) -> Any:
        """Return the RNG used for dynamic threshold sampling.

        Training leaves `threshold_seed=None`, so this uses the process-global
        NumPy RNG and changes across calls. Eval can set `threshold_seed` on a
        copied task to get deterministic dynamic thresholds.
        """
        if self.threshold_seed is None:
            return np.random

        return np.random.default_rng(self.threshold_seed)

    def _empty_cleanup_result(self) -> Mapping[str, torch.Tensor]:
        return {
            "query_anchor_idx": torch.zeros(0, dtype=torch.long),
            "query_target_idx": torch.zeros(0, dtype=torch.long),
            "query_tau": torch.zeros(0, dtype=torch.float32),
            "query_direction": torch.zeros(0, dtype=torch.long),
            "query_y": torch.zeros((0, self.t_surv), dtype=torch.float32),
            "query_weight": torch.zeros((0, self.t_surv), dtype=torch.float32),
        }


def _to_numpy(value: Any) -> np.ndarray:
    """Detach tensors from dataset formatting and expose them as numpy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
