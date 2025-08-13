#!/usr/bin/env python3
"""
Benchmark script to compare old vs new cleanup performance
"""
import time
import torch
import numpy as np

def create_mock_batch(num_pred_points=300, num_tasks=6100, num_bins=20):
    """Create mock batch data for testing"""
    
    # Create random time-to-event data (sparse)
    density = 0.1  # 10% of prediction-task combinations have events
    num_events = int(num_pred_points * num_tasks * density)
    
    # Mock sparse time data
    data = np.random.exponential(scale=100, size=num_events).astype(np.float32)
    indices = np.random.randint(0, num_tasks, size=num_events, dtype=np.int32)
    
    # Create indptr for sparse matrix
    indptr = np.zeros(num_pred_points + 1, dtype=np.int32)
    events_per_pred = np.random.poisson(lam=density * num_tasks, size=num_pred_points)
    events_per_pred = np.minimum(events_per_pred, num_tasks)  # Cap at num_tasks
    indptr[1:] = np.cumsum(events_per_pred)
    
    # Adjust if we have too many events
    if indptr[-1] > num_events:
        # Scale down
        scale_factor = num_events / indptr[-1]
        events_per_pred = (events_per_pred * scale_factor).astype(int)
        indptr[1:] = np.cumsum(events_per_pred)
    
    # Truncate data to match indptr
    actual_events = indptr[-1]
    data = data[:actual_events]
    indices = indices[:actual_events]
    
    # Mock time bins - each task gets different bins
    time_bins = np.zeros((num_tasks, num_bins + 1))
    for i in range(num_tasks):
        # Each task gets exponentially spaced bins
        max_time = 1000 + np.random.exponential(500)
        task_bins = np.logspace(0, np.log10(max_time), num_bins + 1)
        task_bins[0] = 0
        task_bins[-1] = float('inf')
        time_bins[i] = task_bins
    
    # Mock censor times
    censor_time = np.random.exponential(scale=200, size=num_pred_points).astype(np.float32)
    
    return {
        'time_sparse': {
            'data': torch.from_numpy(data),
            'indices': torch.from_numpy(indices),
            'indptr': torch.from_numpy(indptr)
        },
        'censor_time': torch.from_numpy(censor_time)
    }, time_bins

def benchmark_cleanup():
    """Benchmark the cleanup performance"""
    print("Creating mock data...")
    batch, time_bins = create_mock_batch(num_pred_points=300, num_tasks=6100, num_bins=20)
    
    print(f"Data created:")
    print(f"  Prediction points: 300")
    print(f"  Tasks: 6100") 
    print(f"  Time bins per task: 20")
    print(f"  Total events: {len(batch['time_sparse']['data'])}")
    print(f"  Memory usage: ~{300 * 6100 * 20 * 4 / 1024 / 1024:.1f} MB for output tensors")
    
    # Test the vectorized approach
    print("\nTesting vectorized cleanup...")
    
    # Mock the cleanup method components
    num_indices = len(batch['censor_time'])
    num_tasks = 6100
    num_time_bins = 20
    
    def h(a):
        shape = (num_indices, num_tasks)
        a_dict = {k: v.numpy() for k, v in a.items()}
        import scipy.sparse
        s = scipy.sparse.csr_array((a_dict["data"], a_dict["indices"], a_dict["indptr"]), shape=shape)
        return s.toarray()
    
    # Convert sparse to dense (this is the expensive part in real usage)
    print("Converting sparse to dense...")
    start_time = time.time()
    time_dense = h(batch['time_sparse'])
    time_dense = torch.from_numpy(time_dense)
    sparse_to_dense_time = time.time() - start_time
    print(f"Sparse to dense conversion: {sparse_to_dense_time:.3f}s")
    
    # Test vectorized bin assignment
    print("Running vectorized bin assignment...")
    start_time = time.time()
    
    censor_times = batch["censor_time"]
    has_future_event = time_dense != 0
    
    # Convert time_bins to torch tensor for vectorized operations
    time_bins_tensor = torch.from_numpy(time_bins).to(dtype=time_dense.dtype)
    
    # Initialize output tensors
    is_event = torch.zeros(size=(num_indices, num_time_bins, num_tasks), dtype=torch.bool)
    is_censored = torch.zeros(size=(num_indices, num_tasks), dtype=torch.bool)
    
    # Vectorized approach
    time_expanded = time_dense.unsqueeze(1)
    censor_times_expanded = censor_times.unsqueeze(1).unsqueeze(2)
    
    bin_starts = time_bins_tensor[:, :-1].T.unsqueeze(0)
    bin_ends = time_bins_tensor[:, 1:].T.unsqueeze(0)
    
    event_in_bin = (has_future_event.unsqueeze(1) & 
                   (bin_starts <= time_expanded) & 
                   (time_expanded < bin_ends))
    
    censor_in_bin = ((~has_future_event).unsqueeze(1) & 
                    (bin_starts <= censor_times_expanded) & 
                    (censor_times_expanded < bin_ends))
    
    is_event = event_in_bin | censor_in_bin
    is_censored = ~has_future_event
    
    vectorized_time = time.time() - start_time
    print(f"Vectorized bin assignment: {vectorized_time:.3f}s")
    
    # Validation
    bins_per_pred_task = torch.sum(is_event, dim=1)
    exactly_one = torch.sum(bins_per_pred_task == 1)
    total_combinations = num_indices * num_tasks
    
    print(f"\nResults:")
    print(f"  Total prediction-task combinations: {total_combinations:,}")
    print(f"  Combinations with exactly 1 bin marked: {exactly_one:,} ({100*exactly_one/total_combinations:.1f}%)")
    print(f"  Total processing time: {sparse_to_dense_time + vectorized_time:.3f}s")
    print(f"  Rate: {total_combinations / (sparse_to_dense_time + vectorized_time):,.0f} combinations/second")
    
    # Memory usage
    memory_mb = (is_event.numel() + is_censored.numel()) * 4 / 1024 / 1024
    print(f"  Output tensor memory: {memory_mb:.1f} MB")

if __name__ == "__main__":
    try:
        benchmark_cleanup()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("This benchmark requires scipy for sparse matrix operations")
    except Exception as e:
        print(f"Benchmark failed: {e}")