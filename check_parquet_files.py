#!/usr/bin/env python3
"""
Script to check if each directory inside base_path contains a corresponding dir_name.parquet file.
"""

import os
import pathlib
from typing import List, Dict, Tuple

def check_parquet_files(base_path: str) -> Dict[str, Dict]:
    """
    Check if each directory inside base_path contains a corresponding dir_name.parquet file.
    
    Args:
        base_path: Path to the directory containing task directories
        
    Returns:
        Dictionary with results for each directory
    """
    results = {}
    base_path_obj = pathlib.Path(base_path)
    
    if not base_path_obj.exists():
        print(f"❌ Base path does not exist: {base_path}")
        return results
    
    if not base_path_obj.is_dir():
        print(f"❌ Base path is not a directory: {base_path}")
        return results
    
    print(f"🔍 Checking directories in: {base_path}")
    print("-" * 50)
    
    for item in base_path_obj.iterdir():
        dir_name = item.name
        
        if item.is_dir():
            # Check if the corresponding parquet file exists
            parquet_file = item / f"{dir_name}.parquet"
            
            if parquet_file.exists():
                file_size = parquet_file.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                
                results[dir_name] = {
                    'status': 'exists',
                    'file_path': str(parquet_file),
                    'file_size_bytes': file_size,
                    'file_size_mb': file_size_mb,
                    'is_directory': True
                }
                
                print(f"✅ {dir_name}: {dir_name}.parquet exists ({file_size_mb:.2f} MB)")
            else:
                results[dir_name] = {
                    'status': 'missing',
                    'file_path': str(parquet_file),
                    'file_size_bytes': 0,
                    'file_size_mb': 0,
                    'is_directory': True
                }
                
                print(f"❌ {dir_name}: {dir_name}.parquet missing")
        else:
            results[dir_name] = {
                'status': 'not_directory',
                'file_path': str(item),
                'file_size_bytes': 0,
                'file_size_mb': 0,
                'is_directory': False
            }
            
            print(f"⚠️  {dir_name}: Not a directory")
    
    return results

def print_summary(results: Dict[str, Dict]) -> None:
    """Print a summary of the results."""
    if not results:
        print("No results to summarize.")
        return
    
    existing_files = [r for r in results.values() if r['status'] == 'exists']
    missing_files = [r for r in results.values() if r['status'] == 'missing']
    not_directories = [r for r in results.values() if r['status'] == 'not_directory']
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total items checked: {len(results)}")
    print(f"Directories with parquet files: {len(existing_files)}")
    print(f"Directories missing parquet files: {len(missing_files)}")
    print(f"Non-directory items: {len(not_directories)}")
    
    if existing_files:
        total_size = sum(r['file_size_mb'] for r in existing_files)
        avg_size = total_size / len(existing_files)
        print(f"Total size of parquet files: {total_size:.2f} MB")
        print(f"Average file size: {avg_size:.2f} MB")
    
    if missing_files:
        print(f"\n❌ Missing parquet files:")
        for dir_name, result in results.items():
            if result['status'] == 'missing':
                print(f"  - {dir_name}")
    
    if not_directories:
        print(f"\n⚠️  Non-directory items:")
        for dir_name, result in results.items():
            if result['status'] == 'not_directory':
                print(f"  - {dir_name}")

def main():
    """Main function to run the parquet file check."""
    # You can change this path as needed
    base_path = "/user/zj2398/cache/mimic/mimic-3.1-meds/patient_outcome_tasks/task"
    
    print("🔍 Parquet File Checker")
    print("=" * 50)
    
    # Check for parquet files
    results = check_parquet_files(base_path)
    
    # Print summary
    print_summary(results)
    
    # Return results for further processing if needed
    return results

if __name__ == "__main__":
    results = main() 