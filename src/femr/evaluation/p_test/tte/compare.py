"""
Paired bootstrap comparison of C-index between ORA and each baseline.

Every model's tte_results_<TRUNCATION>.pkl was written by tte_test.py using the
same bootstrap indices and the same test row order, so the comparison is paired;
paired_bootstrap_test asserts that ordering matches before computing anything.

Run with: python compare.py
"""
from pathlib import Path
import pickle

from utils import paired_bootstrap_test

BASE = Path("/shared/share_mala/zj2398/mimic/task/phenotype_task/results")
TASKS = ["ami", "masld", "stroke"]
TRUNCATION = 300.0

# Directory name of the model under test, and the baselines it is compared to.
# Adding a model is a single line here: (directory name, display name).
# All entries are the MIMIC transformer runs; `icarefm_730` is the transformer
# run despite the name (the mamba one is `icare_mamba_730`).
ORA_DIR = "transformer_mtpp_730"
BASELINES = [
    ("deephit", "Deephit"),
    ("transformer_ntp_730", "NTP"),
    ("motor_730", "MOTOR"),
    ("icarefm_730", "ICARE-FM"),
    ("transformer_survivehr_730", "SurviveHR"),
]


def load_results(task, model_dir):
    with open(BASE / task / model_dir / f"tte_results_{TRUNCATION}.pkl", "rb") as f:
        return pickle.load(f)


# results_table[display_name][task] = dict from paired_bootstrap_test
results_table = {name: {} for _, name in BASELINES}

for task in TASKS:
    print(f"\n=== Task: {task} ===")
    ora = load_results(task, ORA_DIR)
    for model_dir, name in BASELINES:
        baseline = load_results(task, model_dir)
        results_table[name][task] = paired_bootstrap_test(baseline, ora, name, "ORA")


# === LaTeX table ===
def fmt_cell(r):
    lo, hi = r["ci"]
    star = "*" if r["p_value"] < 0.05 else ""
    return f"[{lo:.4f}, {hi:.4f}]{star}"


print("\n\n=== LaTeX Table (ΔC-index 95% CI: ORA − baseline; * = p<0.05) ===")
print("\\begin{tabular}{l" + "c" * len(TASKS) + "}")
print("\\toprule")
print("Model & " + " & ".join(TASKS) + " \\\\")
print("\\midrule")
for _, name in BASELINES:
    print(name + " & " + " & ".join(fmt_cell(results_table[name][t]) for t in TASKS) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
