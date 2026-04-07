"""
Compare C-index results from any two saved survival result files.

Usage:
"""
from pathlib import Path
import pickle
from utils import paired_bootstrap_test

base = Path("/shared/share_mala/zj2398/mimic/task/phenotype_task/results/")
tasks = ["ami", "masld", "stroke"]
baselines = [("cc-llama", "llama"), ("motor_730", "motor")]
# baselines = [("llama_730", "llama"), ("motor_730", "motor")]
ref_dir, ref_name = "transformer_mtpp_730", "mtpp"

# results_table[baseline_dir][task] = dict from paired_bootstrap_test
results_table = {b[0]: {} for b in baselines}

for task in tasks:
    with open(base/task/"cc-llama/tte_results_300.0.pkl", "rb") as f:
        llama = pickle.load(f)
    with open(base/task/"motor_730/tte_results_300.0.pkl", "rb") as f:
        motor = pickle.load(f)
    with open(base/task/"transformer_mtpp_730/tte_results_300.0.pkl", "rb") as f:
        mtpp = pickle.load(f)
    # with open(base/task/"transformer_ntp_730/tte_results_new.pkl", "rb") as f:
    #     ntp = pickle.load(f)

    # print(mtpp.keys())
    print(f"\n=== Task: {task} ===")
    results_table["cc-llama"][task]  = paired_bootstrap_test(llama, mtpp, "llama", "mtpp")
    results_table["motor_730"][task] = paired_bootstrap_test(motor, mtpp, "motor", "mtpp")


# === LaTeX table ===
def fmt_cell(r):
    lo, hi = r["ci"]
    star = "*" if r["p_value"] < 0.05 else ""
    return f"[{lo:.4f}, {hi:.4f}]{star}"

print("\n\n=== LaTeX Table (ΔC-index 95% CI: mtpp − baseline; * = p<0.05) ===")
print("\\begin{tabular}{l" + "c" * len(tasks) + "}")
print("\\toprule")
print("Model & " + " & ".join(tasks) + " \\\\")
print("\\midrule")
for b_dir, b_name in baselines:
    row = b_name + " & " + " & ".join(fmt_cell(results_table[b_dir][t]) for t in tasks) + " \\\\"
    print(row)
print("\\bottomrule")
print("\\end{tabular}")
