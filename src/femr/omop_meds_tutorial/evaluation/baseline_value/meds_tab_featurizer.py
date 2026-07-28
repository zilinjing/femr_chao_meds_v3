"""MEDS-Tab-style numeric aggregations as a FEMR Featurizer.

Background
----------
The tabular baseline used in the submission builds features with
``femr.featurizers.CountFeaturizer`` under its default settings
(``numeric_value_decile=False``, ``string_value_combination=False``). Under those
defaults, any event carrying a numeric value is dropped from the feature matrix
*entirely* -- not only its value, but its occurrence count -- because
``add_preprocess_data`` routes numeric events into an ``elif`` branch that adds
nothing when the flag is off, so the code never enters ``observed_codes`` and no
column is ever allocated.

This module restores that information by reproducing the MEDS-Tab
(arXiv:2411.00200) aggregation set exactly:

    code/count, value/count, value/sum, value/sum_sqd, value/min, value/max

over nested look-back windows ending at the prediction time. No extra
aggregations (no ``last``, no recency, no derived mean/std) and no per-code
standardisation: the feature specification is the published one, so the
strengthened baseline is not ours to defend.

Design notes
------------
* Windows are **nested** (``7d`` is contained in ``full``), matching MEDS-Tab.
  This differs from ``CountFeaturizer.time_bins``, which are *disjoint*
  consecutive buckets.
* Full-history aggregates use running accumulators. Finite windows use a bounded
  event buffer that is evicted and rescanned at every label, because ``min`` and
  ``max`` cannot be maintained incrementally under eviction. Keeping the largest
  *finite* window short (7 days) is what keeps that rescan cheap.
* Values are emitted **raw**. FEMR's sparse representation fills unemitted
  columns with 0; for raw laboratory values 0 lies outside the physiological
  range of essentially every analyte, so it acts as an out-of-range sentinel that
  a tree separates with a single split, and ``value/count == 0`` encodes
  missingness explicitly in its own column. This is what MEDS-Tab itself does.
* Codes are selected by numeric support (``min_support`` numeric observations
  across the labelled population). For a selected code, ``code/count`` counts
  *all* occurrences, including those with a null numeric value, matching
  MEDS-Tab. Those null-valued occurrences are also counted by ``CountFeaturizer``
  (they reach its ``else`` branch), so they appear in two columns. Harmless for a
  tree; noted here so it is not mistaken for a bug.
"""

from __future__ import annotations

import collections
import datetime
import math
from collections import deque
from typing import Dict, List, Optional, Sequence

import meds_reader

import femr.labelers
from femr.featurizers.core import ColumnValue, Featurizer


class MedsTabNumericFeaturizer(Featurizer):
    """MEDS-Tab numeric aggregations over nested look-back windows.

    Args:
        windows_days: nested look-backs ending at the prediction time, in days.
            ``None`` denotes full history. Default ``(7, None)``.
        min_support: minimum number of numeric observations (across the labelled
            population) for a code to receive a column block.
    """

    AGGS = (
        "code/count",
        "value/count",
        "value/sum",
        "value/sum_sqd",
        "value/min",
        "value/max",
    )

    def __init__(
        self,
        windows_days: Sequence[Optional[int]] = (7, None),
        min_support: int = 100,
    ):
        assert len(windows_days) > 0, "Need at least one window"
        finite = [w for w in windows_days if w is not None]
        assert all(w > 0 for w in finite), "Finite windows must be positive day counts"
        assert len(set(windows_days)) == len(windows_days), "Duplicate windows"

        self.windows: List[Optional[int]] = list(windows_days)
        self.min_support = int(min_support)

        # Populated by encorperate_prepreprocessed_data.
        self.code_to_base: Dict[str, int] = {}
        self.num_columns: int = 0

    # ------------------------------------------------------------------ #
    # Preprocessing: choose which codes get a column block                #
    # ------------------------------------------------------------------ #
    def is_needs_preprocessing(self) -> bool:
        return True

    def get_initial_preprocess_data(self) -> Dict[str, int]:
        # defaultdict(int) rather than a lambda so the state stays picklable
        # across meds_reader worker processes.
        return collections.defaultdict(int)

    def add_preprocess_data(
        self,
        data: Dict[str, int],
        subject: meds_reader.Subject,
        labels: Sequence[femr.labelers.Label],
    ) -> None:
        for event in subject.events:
            if event.numeric_value is not None:
                data[event.code] += 1

    def encorperate_prepreprocessed_data(self, data_elements: List[Dict[str, int]]) -> None:
        counts: Dict[str, int] = collections.defaultdict(int)
        for element in data_elements:
            for code, n in element.items():
                counts[code] += n

        self.code_to_base = {}
        self.num_columns = 0
        block = len(self.AGGS) * len(self.windows)
        for code in sorted(c for c, n in counts.items() if n >= self.min_support):
            self.code_to_base[code] = self.num_columns
            self.num_columns += block

    def get_num_columns(self) -> int:
        return self.num_columns

    # ------------------------------------------------------------------ #
    # Featurization                                                       #
    # ------------------------------------------------------------------ #
    def featurize(
        self,
        subject: meds_reader.Subject,
        labels: Sequence[femr.labelers.Label],
    ) -> List[List[ColumnValue]]:
        finite = [w for w in self.windows if w is not None]
        max_win = datetime.timedelta(days=max(finite)) if finite else None

        # (time, code, value_or_None) for events inside the largest finite window.
        buf: deque = deque()
        # code -> accumulator, over the full history.
        full: Dict[str, list] = {}

        out: List[List[ColumnValue]] = []
        label_idx = 0

        def blank() -> list:
            # code_n, value_n, sum, sum_sqd, min, max
            return [0, 0, 0.0, 0.0, math.inf, -math.inf]

        def push(acc: list, value: Optional[float]) -> None:
            acc[0] += 1
            if value is None:
                return
            acc[1] += 1
            acc[2] += value
            acc[3] += value * value
            if value < acc[4]:
                acc[4] = value
            if value > acc[5]:
                acc[5] = value

        def cols(code: str, window_idx: int, acc: list) -> List[ColumnValue]:
            if acc[0] == 0:
                return []
            off = self.code_to_base[code] + window_idx * len(self.AGGS)
            row = [ColumnValue(off + 0, acc[0])]
            if acc[1] > 0:
                # Value aggregates only exist if at least one value was observed;
                # otherwise these columns stay at the sparse default of 0, which
                # is what MEDS-Tab reports for a code with no numeric readings.
                row.extend(
                    [
                        ColumnValue(off + 1, acc[1]),
                        ColumnValue(off + 2, acc[2]),
                        ColumnValue(off + 3, acc[3]),
                        ColumnValue(off + 4, acc[4]),
                        ColumnValue(off + 5, acc[5]),
                    ]
                )
            return row

        def emit(prediction_time: datetime.datetime) -> List[ColumnValue]:
            if max_win is not None:
                while buf and buf[0][0] < prediction_time - max_win:
                    buf.popleft()

            windowed: Dict[tuple, list] = {}
            for event_time, code, value in buf:
                age = prediction_time - event_time
                for window_idx, window in enumerate(self.windows):
                    if window is None or age > datetime.timedelta(days=window):
                        continue
                    push(windowed.setdefault((code, window_idx), blank()), value)

            row: List[ColumnValue] = []
            for window_idx, window in enumerate(self.windows):
                if window is None:
                    for code, acc in full.items():
                        row.extend(cols(code, window_idx, acc))
            for (code, window_idx), acc in windowed.items():
                row.extend(cols(code, window_idx, acc))
            return row

        for event in subject.events:
            while event.time is not None and event.time > labels[label_idx].prediction_time:
                out.append(emit(labels[label_idx].prediction_time))
                label_idx += 1
                if label_idx >= len(labels):
                    # No later label can see these events; stop early.
                    return out

            if event.code not in self.code_to_base:
                continue

            value = None if event.numeric_value is None else float(event.numeric_value)
            push(full.setdefault(event.code, blank()), value)
            if max_win is not None and event.time is not None:
                buf.append((event.time, event.code, value))

        for label in labels[label_idx:]:
            out.append(emit(label.prediction_time))

        return out

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    def get_column_name(self, column_idx: int) -> str:
        block = len(self.AGGS) * len(self.windows)
        for code, base in self.code_to_base.items():
            if base <= column_idx < base + block:
                offset = column_idx - base
                window = self.windows[offset // len(self.AGGS)]
                agg = self.AGGS[offset % len(self.AGGS)]
                suffix = "full" if window is None else f"{window}d"
                return f"{code}_{agg}_{suffix}"
        raise RuntimeError(f"Could not find name for column {column_idx}")

    def __repr__(self) -> str:
        return (
            f"MedsTabNumericFeaturizer(codes={len(self.code_to_base)}, "
            f"windows={self.windows}, min_support={self.min_support}, "
            f"columns={self.num_columns})"
        )
