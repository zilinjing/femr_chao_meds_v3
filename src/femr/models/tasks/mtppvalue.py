from __future__ import annotations

import abc
import collections
import datetime
import functools
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
import sys
import meds
import meds_reader
import numpy as np
import scipy.sparse
import torch
import warnings
import pickle as pl

import femr.models.config
import femr.models.tokenizer
import femr.ontology
import femr.pat_utils
import femr.stat_utils
import random
import logging

class Task(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_task_config(self) -> femr.models.config.FEMRTaskConfig: ...

    @abc.abstractmethod
    def start_batch(self) -> None: ...

    @abc.abstractmethod
    def start_subject(self, subject: meds_reader.Subject, ontology: Optional[femr.ontology.Ontology]) -> None: ...

    @abc.abstractmethod
    def add_subject_labels(self, subject_label_offsets: List[int]) -> None: ...

    @abc.abstractmethod
    def needs_exact(self) -> bool: ...

    @abc.abstractmethod
    def get_sampled_labels(self, length: int) -> int:
        return length

    @abc.abstractmethod
    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]],
    ) -> int: ...

    @abc.abstractmethod
    def get_batch_data(self) -> Mapping[str, np.ndarray]: ...

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return batch


class MTPPValue_SurvivalCalculator:
    def __init__(
            self, ontology: femr.ontology.Ontology, subject: meds_reader.Subject,
            code_whitelist: Optional[Set[str]] = None
    ):
        self.survival_events = []  # a list of tuples, each tuple is a code, time, and optional value
        self.final_date = subject.events[-1].time
        self.future_times = collections.defaultdict(list)   # a dictionary of lists, key is the code, value is a list of (time, value) tuples

        for event in subject.events:
            if event.time is None:
                continue
            
            # Handle numerical codes differently
            is_numerical = event.numeric_value is not None and event.text_value is None
            
            # For numerical codes: include if value is non-empty and can be float()
            # For non-numerical codes: exclude if has numeric_value or text_value (original logic)
            if is_numerical:
                # Include numerical codes with valid values - use direct code, no ontology needed
                try:
                    float_value = float(event.numeric_value)
                    if code_whitelist is None or event.code in code_whitelist:
                        self.future_times[event.code].append((event.time, float_value))
                        self.survival_events.append((event.code, event.time, float_value))
                except (ValueError, TypeError):
                    # Skip if can't convert to float
                    continue
            # else:
            #     # Original logic for non-numerical codes
            #     if event.numeric_value is not None or event.text_value is not None:
            #         continue
            #     codes = set()
            #     for parent in ontology.get_all_parents(event.code):
            #         if code_whitelist is None or parent in code_whitelist:
            #             codes.add(parent)

            #     for code in codes:
            #         self.future_times[code].append((event.time, None))  # None for non-numerical
            #         self.survival_events.append((code, event.time, None))

        for v in self.future_times.values():
            v.reverse()

        self.survival_events.reverse()

    # Advancing the cursor: removes any past or current events so that only future ones remain in future_times.
    def get_future_events_for_time(
            self, time: datetime.datetime
    ) -> Tuple[datetime.timedelta, Mapping[str, Tuple[datetime.timedelta, Optional[float]]]]:
        while len(self.survival_events) > 0 and self.survival_events[-1][1] <= time:
            code = self.survival_events[-1][0]
            vals = self.future_times[code]
            vals.pop()
            if len(vals) == 0:
                del self.future_times[code]

            self.survival_events.pop()

        delta = self.final_date - time
        # k is the code, v is the list of (time, value) tuples
        # v[-1][0] is the last time in the list, v[-1][1] is the value
        # time is the current time
        # so v[-1][0] - time is the time until the next event of the code
        return (delta, {k: (v[-1][0] - time, v[-1][1]) for k, v in self.future_times.items()})


def _prefit_motor_map(
        subjects: Iterator[meds_reader.Subject], *, tasks: List[str], ontology: femr.ontology.Ontology
) -> Any:
    task_time_stats: List[Any] = [[0, 0, femr.stat_utils.OnlineStatistics()] for _ in range(len(tasks))]
    # Create separate ReservoirSamplers for each task/event type
    task_event_times = [femr.stat_utils.ReservoirSampler(100_000) for _ in range(len(tasks))]
    # Create separate value samplers for numerical tasks
    task_value_stats = [femr.stat_utils.ReservoirSampler(100_000) for _ in range(len(tasks))]
    task_set = set(tasks)

    for subject in subjects:
        calculator = MTPPValue_SurvivalCalculator(ontology, subject, task_set)
        birth = femr.pat_utils.get_subject_birthdate(subject)

        for event, next_event in zip(subject.events, subject.events[1:]):
            # 1) Skip any "birth"‐day events or events with missing times
            if (event.time is None) or (event.time.date() == birth.date()) or (event.time.date() == next_event.time.date()):
                continue
            
            # 2) Ask the calculator: 
            #    - `censor_time`: time until end‐of‐record
            #    - `tte`: dict of per‐code (time_delta, value) tuples
            censor_time, tte = calculator.get_future_events_for_time(event.time)

            if len(tte) == 0:
                continue
            
            for i, task in enumerate(tasks):
                if task in tte:
                    time_delta, value = tte[task]
                    is_censored = False
                else:
                    time_delta = censor_time
                    value = None
                    is_censored = True

                if is_censored:
                    task_time_stats[i][0] += 1
                else:
                    # Add to the specific task's event time sampler
                    task_event_times[i].add(time_delta.total_seconds(), 1)
                    task_time_stats[i][1] += 1
                    
                    # Add value to sampler if it's a numerical task
                    if value is not None:
                        task_value_stats[i].add(value, 1)
                
                task_time_stats[i][2].add(1, time_delta.total_seconds())
    
    
    print(f"prefit_motor_map is done")
    return (task_event_times, task_time_stats, task_value_stats)


def _prefit_motor_agg(first: Any, second: Any) -> Any:
    # first/second is (task_event_times, task_time_stats, task_value_stats)
    # for task_event_times, we combine each task's ReservoirSampler separately
    # for task_time_stats, we add the number of events/censoring times, and combine the two groups of elements
    for a, b in zip(first[1], second[1]):
        a[0] += b[0]
        a[1] += b[1]
        a[2].combine(b[2])
    # Combine each task's event times separately
    for i in range(len(first[0])):
        first[0][i].combine(second[0][i])
    # Combine each task's value stats separately
    for i in range(len(first[2])):
        first[2][i].combine(second[2][i])
    return first

def keep_unique_bins(value_samples, num_bins):
    unique_bins = np.unique(value_samples)
    unique_bins_augmented = np.concatenate([unique_bins,np.array((num_bins+1-len(unique_bins))*[float("inf")])])
    mask = np.concatenate([np.ones(len(unique_bins)),np.zeros(num_bins+1-len(unique_bins))])
    return unique_bins_augmented, mask
    
class MTPPValue_Task(Task):
    @classmethod
    def fit_pretraining_task_info(
            cls,
            db: meds_reader.SubjectDatabase,
            tokenizer: femr.models.tokenizer.HierarchicalTokenizer,
            num_tasks: int,
            num_final_tasks: int,
            num_bins: int,
            final_layer_size: int,
            value_count_threshold: int,
            codes_to_skip: List[str] = None,
            num_value_bins: int = 10
    ) -> MTPPValue_Task:
        tasks = []
        print("using mtpp_value")
        for dict_entry in tokenizer.dictionary["vocab"]:
            if dict_entry["type"] == "code":
                # Skip the codes that are in the codes_to_skip
                if codes_to_skip and dict_entry["code_string"] in codes_to_skip:
                    continue
                tasks.append(dict_entry["code_string"])
                if len(tasks) == num_tasks:
                    break

        if len(tasks) < num_tasks:
            warnings.warn(f"Could not find enough tasks in the provided tokenizer {len(tasks)}")

        print(f"Processing {len(tasks)} tasks for fitting")
        print(f"Using {num_bins} time bins and {num_value_bins} value bins")
        
        # Apply parallel processing - ORDER IS PRESERVED by index
        task_length_samples, stats, task_value_samples = functools.reduce(
            _prefit_motor_agg, db.map(functools.partial(_prefit_motor_map, tasks=tasks, ontology=tokenizer.ontology))
        )

        # First pass: keep only numerical tasks with observed values.
        numerical_tasks_info = []  # List of (index, task, task_stats, rate, value_sample_count)

        full_task_path = "/data2/processed_datasets/zj2398/femr/mimic/mtpp_T_8_V_5/mtpp_task.pkl"
        with open(full_task_path,"rb") as f:
            mtpp_task = pl.load(f)
        full_codes = set(mtpp_task.numerical_task_to_index_map.keys())
        print(f"it should have {len(full_codes)} codes")

        for i, (task, task_stats) in enumerate(zip(tasks, stats)):
            frac_events = task_stats[1] / (task_stats[0] + task_stats[1])
            rate = frac_events / task_stats[2].mean()

            if rate == 0:
                print(f"Ran into task of rate 0: {task}, frac_events={frac_events}, "
                    f"censored={task_stats[0]}, events={task_stats[1]}, mean_time={task_stats[2].mean()}")
                continue

            value_sample_count = len(task_value_samples[i].samples)
            
            if value_sample_count > 0:
                # Numerical task - stricter filtering
                # if frac_events < 1 / 1000 or value_sample_count < value_count_threshold:
                #     print(f"Filtered out numerical task {task}: frac_events={frac_events}, "
                #         f"value_count={value_sample_count}")
                #     continue
                if task not in full_codes:
                    continue
                numerical_tasks_info.append((i, task, task_stats, rate, value_sample_count))

        # Apply num_final_tasks limit
        if len(numerical_tasks_info) > num_final_tasks:
            print(
                f"Limiting numerical tasks from {len(numerical_tasks_info)} to "
                f"{num_final_tasks} to stay within num_final_tasks={num_final_tasks}"
            )
            numerical_tasks_info = numerical_tasks_info[:num_final_tasks]

        total_filtered = len(numerical_tasks_info)
        print(f"After filtering: {len(numerical_tasks_info)} numerical tasks")
        if total_filtered == 0:
            raise ValueError("MTPPValue_Task.fit_pretraining_task_info found no eligible numerical tasks.")

        # Second pass: create bins for selected tasks in the correct order
        numerical_time_bins = []
        value_bins = []
        value_valid_mask = []
        numerical_task_list = []
        task_data = []

        # Process numerical tasks first
        for i, task, task_stats, rate, value_sample_count in numerical_tasks_info:
            task_data.append((task, rate, task_stats[0], task_stats[1], task_stats[2].mean()))
            
            # Create time bins for numerical task
            numerical_task_time_bins = np.percentile(
                task_length_samples[i].samples, np.linspace(0, 100, num_bins + 1)
            )
            numerical_task_time_bins[0] = 0
            numerical_task_time_bins[-1] = float("inf")
            numerical_time_bins.append(list(numerical_task_time_bins))

            # Create value bins
            task_value_bins = np.percentile(
                task_value_samples[i].samples, np.linspace(0, 100, num_value_bins + 1)
            )
            task_value_bins[0] = float("-inf")
            task_value_bins[-1] = float("inf")
            unique_bins, mask = keep_unique_bins(task_value_bins, num_value_bins)

            value_bins.append(unique_bins)
            value_valid_mask.append(mask)
            numerical_task_list.append(task)
            
            print(f"Created {int(sum(mask))} valid value bins for numerical task {task} "
                f"from {value_sample_count} samples")

        # Convert to arrays
        numerical_time_bins = np.array(numerical_time_bins) if numerical_time_bins else np.array([]).reshape(0, num_bins + 1)
        value_bins = np.array(value_bins) if value_bins else np.array([]).reshape(0, num_value_bins + 1)
        value_valid_mask = np.array(value_valid_mask) if value_valid_mask else np.array([]).reshape(0, num_value_bins + 1)
        
        print(f"Final task count: {len(task_data)}")
        print(f"Numerical tasks: {len(numerical_task_list)}")
        print(f"Numerical time bins shape: {numerical_time_bins.shape}")
        print(f"Value bins shape: {value_bins.shape}")

        return MTPPValue_Task(
            task_data, 
            numerical_time_bins, 
            final_layer_size, 
            value_bins, 
            value_valid_mask, 
            num_value_bins, 
            numerical_task_list
        )

    def __init__(self, pretraining_task_info: List[Tuple[str, float]], numerical_time_bins: np.ndarray, final_layer_size: int,
                 value_bins: Optional[np.ndarray] = None, value_valid_mask: Optional[np.ndarray] = None, num_value_bins: int = 10, numerical_task_list: List[str] = None):
        

        self.pretraining_task_info = pretraining_task_info
        
        # Handle both numpy array and list (from config deserialization)
         # Now a 2D array: [num_tasks, num_bins+1]
        # self.non_numerical_time_bins = non_numerical_time_bins
        self.numerical_time_bins = numerical_time_bins
        self.final_layer_size = final_layer_size
        
        # Value bins for numerical tasks
        self.value_bins = np.array(value_bins) if not isinstance(value_bins, np.ndarray) else value_bins   # eg: [[-float("inf"),1,2,3,4,5,6,7,8,float("inf"),float("inf")],
        self.value_valid_mask = np.array(value_valid_mask) if not isinstance(value_valid_mask, np.ndarray) else value_valid_mask    # eg: [[1,1,1,1,1,1,1,1,1,1,0]
        self.num_value_bins = num_value_bins
        self.numerical_task_list = numerical_task_list

        self.pretraining_task_codes = set()
        self.numerical_task_to_index_map = {}  # Only for numerical codes
        
        # the self.pretraining_task_info is a list of tuples, each tuple is a task name, rate, num_events, num_censored, mean_time
        numerical_idx = 0
        for i, task in enumerate(self.pretraining_task_info):
            task_name = task[0]
            self.pretraining_task_codes.add(task_name)
            
            if task_name in self.numerical_task_list:
                self.numerical_task_to_index_map[task_name] = numerical_idx
                numerical_idx += 1

        assert len(self.numerical_task_to_index_map) == len(self.numerical_task_list), f"len(self.numerical_task_to_index_map) = {len(self.numerical_task_to_index_map)}, len(self.numerical_task_list) = {len(self.numerical_task_list)}"
        assert len(self.pretraining_task_info) == len(self.numerical_task_list), (
            "MTPPValue_Task should only contain numerical tasks. "
            f"Found {len(self.pretraining_task_info)} tasks but {len(self.numerical_task_list)} numerical tasks."
        )



    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="mtpp_value",
            task_kwargs=dict(
                pretraining_task_info=self.pretraining_task_info,
                numerical_task = list(self.numerical_task_to_index_map.keys()), # Only for numerical codes
                value_bins=self.value_bins.tolist() if isinstance(self.value_bins, np.ndarray) else self.value_bins,
                numerical_task_time_bins=self.numerical_time_bins.tolist() if isinstance(self.numerical_time_bins, np.ndarray) else self.numerical_time_bins,
                final_layer_size=self.final_layer_size,
            ),
        )

    def start_subject(self, subject: meds_reader.Subject, ontology: Optional[femr.ontology.Ontology]) -> None:
        assert ontology
        self.calculator = MTPPValue_SurvivalCalculator(ontology, subject, self.pretraining_task_codes)

        self.per_subject_censor_time: List[float] = []
        self.per_subject_value_sparse: Dict[str, List[float]] = {
            "time": [],  # time of the event
            "value": [],  # value of event  
            "indices": [],  # code for lab value prediction in numerical vocabulary
            "indptr": [0],  # index of each time point
        }

    def needs_exact(self) -> bool:
        return False

    def get_sampled_labels(self, length: int) -> int:
        desired_labels = max(5, length // 10)
        return desired_labels

    def start_batch(self) -> None:
        self.censor_time: List[float] = []
        
        self.value_sparse: Dict[str, List[float]] = {
            "time": [],  # time of the event
            "value": [],  # value of event
            "indices": [],  # code for lab value prediction in numerical vocabulary
            "indptr": [0],  # index of each time point
        }

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        """Add per-subject labels to the global task labels."""
        self.censor_time.extend([self.per_subject_censor_time[i] for i in subject_label_offsets])

        for index in subject_label_offsets:
            value_start = int(self.per_subject_value_sparse["indptr"][index])
            value_end = int(self.per_subject_value_sparse["indptr"][index + 1])
            self.value_sparse["time"].extend(self.per_subject_value_sparse["time"][value_start:value_end])
            self.value_sparse["value"].extend(self.per_subject_value_sparse["value"][value_start:value_end])
            self.value_sparse["indices"].extend(self.per_subject_value_sparse["indices"][value_start:value_end])
            self.value_sparse["indptr"].append(len(self.value_sparse["indices"]))
            

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
            actually_add: bool = True,
    ) -> int:
        if next_date is None or next_date == current_date:
            return 0
        
        if not actually_add:
            return 1

        censor_time, tte = self.calculator.get_future_events_for_time(current_date)

        if len(tte) == 0:
            return 0


        censor_seconds = censor_time.total_seconds()
        self.per_subject_censor_time.append(censor_seconds)

        for event_name, (time_delta, value) in tte.items():
            seconds = time_delta.total_seconds()

            if value is not None and event_name in self.numerical_task_to_index_map:
                numerical_j = self.numerical_task_to_index_map[event_name]
                self.per_subject_value_sparse["time"].append(seconds)
                self.per_subject_value_sparse["value"].append(value)
                self.per_subject_value_sparse["indices"].append(numerical_j)

        self.per_subject_value_sparse["indptr"].append(len(self.per_subject_value_sparse["time"]))

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        def h(a, dtype):
            result = {
                "time": np.array(a["time"], dtype=dtype),
                "indices": np.array(a["indices"], dtype=np.int32),
                "indptr": np.array(a["indptr"], dtype=np.int32),
            }
            # Add value field for value_sparse
            if "value" in a:
                result["value"] = np.array(a["value"], dtype=dtype)
            return result

        return {
            "censor_time": np.array(self.censor_time, dtype=np.float32),
            "value_sparse" : h(self.value_sparse, dtype=np.float32),
        }

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        Convert sparse numerical time/value labels into dense tensors for loss calculation.
        """
        assert self.numerical_time_bins.shape[0] == len(self.numerical_task_to_index_map), (
            "numerical_time_bins and numerical_task_to_index_map disagree on task count: "
            f"{self.numerical_time_bins.shape[0]} vs {len(self.numerical_task_to_index_map)}"
        )

        num_time_bins = self.numerical_time_bins.shape[1] - 1
        num_indices = len(batch["censor_time"])
        num_numerical_tasks = len(self.numerical_task_to_index_map)
        if num_numerical_tasks == 0:
            raise ValueError("MTPPValue_Task.cleanup requires at least one numerical task.")

        def sparse_to_dense(sparse_data, num_tasks):
            """Convert sparse data to dense tensor with proper edge case handling"""
            shape = (num_indices, num_tasks)
            sparse_dict = {k: v.numpy() for k, v in sparse_data.items()}
            
            # Handle edge case: empty sparse data
            if len(sparse_dict.get("time", [])) == 0 or len(sparse_dict.get("indices", [])) == 0:
                if "value" in sparse_dict:
                    time_dense = torch.zeros(shape, dtype=torch.float32)
                    value_dense = torch.full(shape, float('nan'), dtype=torch.float32)
                    return time_dense, value_dense
                else:
                    return torch.zeros(shape, dtype=torch.float32)
            
            # Handle the case where we have 'value' field in sparse data
            if "value" in sparse_dict:
                # For value sparse data, we need to return both time and value
                time_sparse = scipy.sparse.csr_array(
                    (sparse_dict["time"], sparse_dict["indices"], sparse_dict["indptr"]), 
                    shape=shape
                )
                
                # Initialize value tensor with NaN (None equivalent) instead of 0
                # Values can range from -inf to +inf, so 0 is not appropriate default
                value_dense = torch.full((num_indices, num_tasks), float('nan'), dtype=torch.float32)
                
                # Fill in actual values only where they exist - handle edge cases
                if len(sparse_dict["indptr"]) > 1:
                    for i in range(len(sparse_dict["indptr"]) - 1):
                        start = sparse_dict["indptr"][i]
                        end = sparse_dict["indptr"][i + 1]
                        if end > start and start < len(sparse_dict["indices"]) and end <= len(sparse_dict["value"]):
                            indices = sparse_dict["indices"][start:end]
                            values = sparse_dict["value"][start:end]
                            if len(indices) > 0 and len(values) > 0:
                                valid_indices = indices[indices < num_tasks]  # Bounds check
                                valid_values = values[:len(valid_indices)]
                                if len(valid_indices) > 0:
                                    value_dense[i, valid_indices] = torch.tensor(valid_values, dtype=torch.float32)
                
                return torch.from_numpy(time_sparse.toarray()), value_dense
            else:
                # Regular time sparse data
                sparse_matrix = scipy.sparse.csr_array(
                    (sparse_dict["time"], sparse_dict["indices"], sparse_dict["indptr"]), 
                    shape=shape
                )
                return torch.from_numpy(sparse_matrix.toarray())

        # Convert to dense tensors
        censor_times = batch["censor_time"]  # [pred_points]
        result = {}

        if num_numerical_tasks > 0 and "value_sparse" in batch:
            time_numerical, values_numerical = sparse_to_dense(batch["value_sparse"], num_numerical_tasks)
            
            # Create time and value bins using matrix operations - returns separate components
            bins_dict = self._create_time_value_bins(
                time_numerical, values_numerical, censor_times, num_time_bins
            )
            
            result.update({
                "numerical_time_event_bin": bins_dict['time_event_in_bin'],      # For event loss computation
                "numerical_time_censor_bin": bins_dict['time_censor_in_bin'],    # For censor loss computation  
                "numerical_value_event_bin": bins_dict['value_event_in_bin'],    # For value binning
                # "numerical_is_event": bins_dict['time_is_event'],         # Combined time events
                "numerical_is_censored": bins_dict['time_is_censored'],    # Censoring status
                "value_valid_mask": self.value_valid_mask
            })
        
        return result


    def _create_time_value_bins(self, time_numerical, values_numerical, censor_times, num_time_bins):
        """
        Create time and value bins for numerical tasks using matrix operations.
        time_numerical: [pred_points, numerical_tasks] time value in each position otherwise 0
        values_numerical: [pred_points, numerical_tasks] value in each position otherwise None
        censor_times: [pred_points]
        num_time_bins: int

        This mirrors _create_time_bins logic but handles both time and value dimensions.
        Returns separate event_in_bin and censor_in_bin matrices for flexible loss computation.
        """
        num_indices, num_numerical_tasks = time_numerical.shape
        device = time_numerical.device
        
        # Determine future events: has_future_event = (time != 0) & (~isnan(value))
        # has_future_event = (time_numerical != 0) & (~torch.isnan(values_numerical))

        # print(time_numerical)
        # print(values_numerical)
        # print(time_numerical.shape)
        # print(values_numerical.shape)
        match = ((time_numerical != 0) == (~torch.isnan(values_numerical))).all()
        if not match:
            print(torch.sum(time_numerical != 0))
            print(torch.sum(~torch.isnan(values_numerical)))
            print(time_numerical.shape)
            print(values_numerical.shape)
        assert ((time_numerical != 0) == (~torch.isnan(values_numerical))).all(), f"time_numerical != 0 = {time_numerical != 0}, ~torch.isnan(values_numerical) = {~torch.isnan(values_numerical)}"
        has_future_event = (time_numerical != 0)
        
        # Convert time_bins to torch tensor for vectorized operations
        time_bins_tensor = torch.from_numpy(self.numerical_time_bins).to(device=device, dtype=time_numerical.dtype) # [num_tasks, num_bins+1 9]
        
        # === TIME BIN OPERATIONS ===
        # Expand dimensions for broadcasting: same logic as _create_time_bins
        time_expanded = time_numerical.unsqueeze(1)  # [pred_points, 1, numerical_tasks]
        censor_times_expanded = censor_times.unsqueeze(1).unsqueeze(2)  # [pred_points, 1, 1]
        
        # Extract start and end of each time bin
        time_bin_starts = time_bins_tensor[:, :-1].T.unsqueeze(0)  # [1, num_bins, numerical_tasks]  
        time_bin_ends = time_bins_tensor[:, 1:].T.unsqueeze(0)     # [1, num_bins, numerical_tasks]
        
        # Time events: which time bin each event falls into
        # Shape: [pred_points, num_time_bins, numerical_tasks]
        time_event_in_bin = (has_future_event.unsqueeze(1) & 
                            (time_bin_starts <= time_expanded) & 
                            (time_expanded < time_bin_ends))
        
        # Time censoring: which time bin each censor time falls into  
        time_censor_in_bin = ((~has_future_event).unsqueeze(1) & 
                             (time_bin_starts <= censor_times_expanded) & 
                             (censor_times_expanded < time_bin_ends))
        
        # === VALUE BIN OPERATIONS ===
        # Create value bins for each numerical task using same vectorized approach

        value_bins = torch.from_numpy(self.value_bins).to(device=device, dtype=values_numerical.dtype) # [num_tasks, num_value_bins+1 11]
        value_expanded = values_numerical.unsqueeze(1) # [pred_points, 1, numerical_tasks]
        

        value_bins_starts = value_bins[:, :-1].T.unsqueeze(0) # [1, num_value_bins, numerical_tasks]
        value_bins_ends = value_bins[:, 1:].T.unsqueeze(0) # [1, num_value_bins, numerical_tasks]
        value_event_in_bin = (has_future_event.unsqueeze(1) & 
                             (value_bins_starts <= value_expanded) & 
                             (value_expanded < value_bins_ends))

        time_bin_counts = torch.sum(time_event_in_bin | time_censor_in_bin, dim=1)
        assert torch.all(time_bin_counts == 1), "Each prediction/task pair should map to exactly one time bin."

        if torch.any(has_future_event):
            value_bin_counts = torch.sum(value_event_in_bin, dim=1)
            assert torch.all(value_bin_counts[has_future_event] == 1), (
                "Each numerical event should map to exactly one value bin."
            )

        time_is_censored = ~has_future_event  # [pred_points, numerical_tasks]
        
        return {
            'time_event_in_bin': time_event_in_bin,      # [pred_points, time_bins, numerical_tasks]
            'time_censor_in_bin': time_censor_in_bin,    # [pred_points, time_bins, numerical_tasks]  
            'value_event_in_bin': value_event_in_bin,    # [pred_points, value_bins, numerical_tasks]
            'time_is_censored': time_is_censored         # [pred_points, numerical_tasks]
        }
