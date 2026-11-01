'''
usage

result_path = "/data/results/zj2398/fomoh/mimic/results"

python rename.py /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/results -o cumc_results.csv
'''

import os
import json
import re
import pandas as pd
from pathlib import Path

def rename_files_and_dirs(base_dir):
    """Remove '_linear_probing_' from all file and directory names."""
    base_path = Path(base_dir)
    
    for disease_dir in base_path.iterdir():
        if not disease_dir.is_dir():
            continue
        
        # Collect all items to rename (files and dirs)
        items_to_rename = []
        for item in disease_dir.iterdir():
            if "_code_8k_float32_" in item.name:
                new_name = item.name.replace("_code_8k_float32_", "_")
                items_to_rename.append((item, disease_dir / new_name))
        
        # Rename (do directories last to avoid path issues)
        # Sort: files first (has extension), then directories
        items_to_rename.sort(key=lambda x: x[0].is_dir())
        
        for old_path, new_path in items_to_rename:
            print(f"Renaming: {old_path.name} -> {new_path.name}")
            old_path.rename(new_path)

def extract_results(base_dir):
    """Extract ROC-AUC scores from the largest-numbered subdirectories."""
    base_path = Path(base_dir)
    results = {}  # {disease: {model: (mean, std)}}
    
    for disease_dir in base_path.iterdir():
        if not disease_dir.is_dir():
            continue
        
        disease_name = disease_dir.name
        results[disease_name] = {}
        
        # Find all model subdirectories (after renaming, pattern is model_*)
        model_dirs = {}  # {model_name: {num: path}}
        
        for item in disease_dir.iterdir():
            if item.is_dir():
                # Match pattern: llama_* or mamba_* (after renaming)
                # Or original: llama_linear_probing_* or mamba_linear_probing_*
                match = re.match(r'^(llama|mamba)(?:_code_8k_float32)?_(\d+)$', item.name)
                if match:
                    model_name = match.group(1)
                    num = int(match.group(2))
                    if model_name not in model_dirs:
                        model_dirs[model_name] = {}
                    model_dirs[model_name][num] = item
        
        # For each model, find the largest numbered directory
        for model_name, num_paths in model_dirs.items():
            if not num_paths:
                continue
            
            max_num = max(num_paths.keys())
            target_dir = num_paths[max_num]
            
            # Read results_boot.json
            json_path = target_dir / "results_boot.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                mean_auc = data.get("mean_roc_auc_score", None)
                std_auc = data.get("std_roc_auc_score", None)
                results[disease_name][model_name] = (mean_auc, std_auc)
                print(f"{disease_name}/{model_name} (n={max_num}): {mean_auc:.4f} ± {std_auc:.4f}")
            else:
                print(f"Warning: {json_path} not found")
    
    return results

def create_csv(results, output_path):
    """Create CSV with rows=models, columns=diseases, values=mean(std)."""
    # Get all diseases and models
    diseases = sorted(results.keys())
    models = set()
    for disease_data in results.values():
        models.update(disease_data.keys())
    models = sorted(models)
    
    # Build dataframe
    data = {}
    for disease in diseases:
        col = []
        for model in models:
            if model in results[disease]:
                mean, std = results[disease][model]
                col.append(f"{mean:.4f}({std:.4f})")
            else:
                col.append("")
        data[disease] = col
    
    df = pd.DataFrame(data, index=models)
    df.index.name = "model"
    df.to_csv(output_path)
    print(f"\nSaved to {output_path}")
    return df

def main(base_dir, output_csv="roc_auc_results.csv", do_rename=True):
    # Step 1: Rename files/directories
    if do_rename:
        print("=== Renaming files and directories ===")
        rename_files_and_dirs(base_dir)
        print()
    
    # Step 2: Extract results
    print("=== Extracting results ===")
    results = extract_results(base_dir)
    
    # Step 3: Create CSV
    print("\n=== Creating CSV ===")
    df = create_csv(results, output_csv)
    print(df)
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory containing disease subdirectories")
    parser.add_argument("-o", "--output", default="roc_auc_results.csv", help="Output CSV path")
    parser.add_argument("--no-rename", action="store_true", help="Skip renaming step")
    args = parser.parse_args()
    
    main(args.base_dir, args.output, do_rename=not args.no_rename)