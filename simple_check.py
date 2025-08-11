#!/usr/bin/env python3
"""
Simple one-liner to check parquet files in directories.
Use this in your notebook or as a quick script.
"""

import os
from pathlib import Path

# Simple one-liner version
def quick_check(base_path):
    """Quick check for parquet files in directories."""
    base = Path(base_path)
    for item in base.iterdir():
        if item.is_dir():
            parquet_file = item / f"{item.name}.parquet"
            status = "✅" if parquet_file.exists() else "❌"
            size_mb = parquet_file.stat().st_size / (1024*1024) if parquet_file.exists() else 0
            print(f"{status} {item.name}: {item.name}.parquet {'exists' if parquet_file.exists() else 'missing'} ({size_mb:.2f} MB)" if parquet_file.exists() else f"{status} {item.name}: {item.name}.parquet missing")
        else:
            print(f"⚠️  {item.name}: Not a directory")

# Usage example:
if __name__ == "__main__":
    base_path = "/user/zj2398/cache/mimic/mimic-3.1-meds/patient_outcome_tasks/task"
    quick_check(base_path) 