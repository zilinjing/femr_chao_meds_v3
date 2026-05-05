"""
Per-task mean and bootstrap SE for each model on MIMIC-IV.

Layout of each panel dict:
    panel_label : str
    metric      : str                              # axis label
    tasks       : list[str]                        # task names within this panel
    methods     : list[str]                        # method names present in this panel
    always_show : list[str]                        # always plotted, even when filtering
    mean        : np.ndarray (n_methods, n_tasks)  # per-task point estimates
    std         : np.ndarray (n_methods, n_tasks)  # per-task bootstrap SE
    colors      : list[str]                        # auto-built from COLORS

Notes on which models appear in which panel
-------------------------------------------
* Classification (6 tasks): full 13-model set incl. XGBoost, Transformer-Numerical, CC-Mamba.
* TTE           (3 tasks): 11 models -- excludes CC-Mamba and Transformer-Numerical
                                       (not in the source table).
* Regression    (4 tasks): 12 models -- excludes XGBoost, DeepHit, Transformer-Numerical.
                                       Most-Recent has *negative* per-task R^2 on PaO2 and
                                       Glucose; this is not a typo.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
# All three task-specific baselines share one gray color so the legend can
# merge them into a single "Task-specific Baseline" entry.
COLORS = {
    # Task-specific baselines (merged in legend)
    'XGBoost':                '#9E9E9E',
    'DeepHit':                '#9E9E9E',
    'Most-Recent':            '#9E9E9E',
    # Foundation-model baselines
    'MOTOR':                  '#C7B8EA',  # lavender
    'CC-Llama':               '#F4C28C',  # pale orange
    'CC-Mamba':               '#F7E08E',  # pale yellow
    # Transformer family
    'Transformer-NTP':        '#B7D4EC',
    'Transformer-TPP':        '#7AA9D6',
    'Transformer-Numerical':  '#D5E5F2',
    'Transformer-SurvivEHR':  '#CCC0DC',
    'Transformer-ICareFM':    '#E8C9A0',
    'Transformer-ORA':        '#1F4E79',  # bold (ours)
    # Mamba family
    'Mamba-NTP':              '#BFE0BF',
    'Mamba-TPP':              '#7DBE7D',
    'Mamba-ORA':              '#1F6B2E',  # bold (ours)
}

TASK_SPECIFIC_BASELINES = {'XGBoost', 'DeepHit', 'Most-Recent'}
TASK_SPECIFIC_LABEL = 'Task-specific Baseline'

HIGHLIGHT_METHODS = {'Transformer-ORA', 'Mamba-ORA'}


# ---------------------------------------------------------------------------
# Classification (AUROC) -- 6 tasks, 13 models
# ---------------------------------------------------------------------------
data_classification = {
    'panel_label': 'Binary Classification',
    'metric': 'AUROC',
    'tasks': ['Readmission', 'LOS', 'Mortality', 'AMI', 'MASLD', 'Stroke'],
    'always_show': ['XGBoost', 'MOTOR'],
    'methods': [
        'XGBoost',
        'CC-Llama',
        'CC-Mamba',
        'MOTOR',
        'Transformer-NTP',
        'Transformer-TPP',
        'Transformer-Numerical',
        'Transformer-SurvivEHR',
        'Transformer-ICareFM',
        'Transformer-ORA',
        'Mamba-NTP',
        'Mamba-TPP',
        'Mamba-ORA',
    ],
    'mean': np.array([
        [0.726, 0.748, 0.882, 0.807, 0.700, 0.708],   # XGBoost
        [0.730, 0.781, 0.901, 0.796, 0.663, 0.696],   # CC-Llama
        [0.731, 0.779, 0.896, 0.783, 0.661, 0.665],   # CC-Mamba
        [0.743, 0.833, 0.954, 0.825, 0.703, 0.710],   # MOTOR
        [0.728, 0.808, 0.926, 0.810, 0.691, 0.691],   # Transformer-NTP
        [0.744, 0.839, 0.963, 0.818, 0.705, 0.700],   # Transformer-TPP
        [0.736, 0.802, 0.955, 0.816, 0.697, 0.686],   # Transformer-Numerical
        [0.714, 0.734, 0.898, 0.803, 0.682, 0.603],   # Transformer-SurvivEHR
        [0.719, 0.796, 0.935, 0.802, 0.659, 0.652],   # Transformer-ICareFM
        [0.745, 0.841, 0.965, 0.827, 0.714, 0.711],   # Transformer-ORA
        [0.732, 0.813, 0.933, 0.814, 0.679, 0.719],   # Mamba-NTP
        [0.741, 0.809, 0.934, 0.822, 0.709, 0.666],   # Mamba-TPP
        [0.747, 0.812, 0.939, 0.832, 0.722, 0.660],   # Mamba-ORA
    ]),
    'std': np.array([
        [0.003, 0.003, 0.005,  0.008, 0.020, 0.023],  # XGBoost
        [0.003, 0.003, 0.005,  0.010, 0.020, 0.025],  # CC-Llama
        [0.003, 0.003, 0.006,  0.009, 0.019, 0.024],  # CC-Mamba
        [0.002, 0.002, 0.0023, 0.008, 0.021, 0.019],  # MOTOR
        [0.003, 0.002, 0.004,  0.008, 0.020, 0.025],  # Transformer-NTP
        [0.004, 0.002, 0.003,  0.007, 0.018, 0.020],  # Transformer-TPP
        [0.003, 0.002, 0.003,  0.006, 0.016, 0.021],  # Transformer-Numerical
        [0.003, 0.003, 0.005,  0.007, 0.016, 0.022],  # Transformer-SurvivEHR
        [0.004, 0.003, 0.005,  0.007, 0.021, 0.023],  # Transformer-ICareFM
        [0.003, 0.002, 0.002,  0.007, 0.017, 0.018],  # Transformer-ORA
        [0.004, 0.002, 0.004,  0.008, 0.020, 0.022],  # Mamba-NTP
        [0.003, 0.003, 0.004,  0.007, 0.020, 0.017],  # Mamba-TPP
        [0.003, 0.002, 0.004,  0.007, 0.020, 0.022],  # Mamba-ORA
    ]),
}


# ---------------------------------------------------------------------------
# Time-to-Event (C-index) -- 3 tasks, 11 models
#   (CC-Mamba and Transformer-Numerical are NOT in the source table)
# ---------------------------------------------------------------------------
data_tte = {
    'panel_label': 'Time-to-Event',
    'metric': 'C-index',
    'tasks': ['AMI', 'MASLD', 'Stroke'],
    'always_show': ['DeepHit', 'MOTOR'],
    'methods': [
        'DeepHit',
        'CC-Llama',
        'MOTOR',
        'Transformer-NTP',
        'Transformer-TPP',
        'Transformer-SurvivEHR',
        'Transformer-ICareFM',
        'Transformer-ORA',
        'Mamba-NTP',
        'Mamba-TPP',
        'Mamba-ORA',
    ],
    'mean': np.array([
        [0.651, 0.560, 0.574],   # DeepHit
        [0.810, 0.688, 0.843],   # CC-Llama
        [0.811, 0.689, 0.832],   # MOTOR
        [0.796, 0.706, 0.702],   # Transformer-NTP
        [0.856, 0.710, 0.886],   # Transformer-TPP
        [0.756, 0.667, 0.832],   # Transformer-SurvivEHR
        [0.757, 0.641, 0.809],   # Transformer-ICareFM
        [0.833, 0.723, 0.890],   # Transformer-ORA
        [0.773, 0.637, 0.743],   # Mamba-NTP
        [0.851, 0.698, 0.819],   # Mamba-TPP
        [0.856, 0.664, 0.849],   # Mamba-ORA
    ]),
    'std': np.array([
        [0.029, 0.024, 0.085],   # DeepHit
        [0.029, 0.022, 0.038],   # CC-Llama
        [0.030, 0.020, 0.041],   # MOTOR
        [0.030, 0.018, 0.056],   # Transformer-NTP
        [0.028, 0.020, 0.030],   # Transformer-TPP
        [0.032, 0.019, 0.044],   # Transformer-SurvivEHR
        [0.035, 0.025, 0.042],   # Transformer-ICareFM
        [0.030, 0.020, 0.027],   # Transformer-ORA
        [0.030, 0.018, 0.064],   # Mamba-NTP
        [0.028, 0.020, 0.042],   # Mamba-TPP
        [0.032, 0.020, 0.034],   # Mamba-ORA
    ]),
}


# ---------------------------------------------------------------------------
# Regression (R^2) -- 4 tasks, 12 models
#   (XGBoost, DeepHit, Transformer-Numerical NOT in the source table.
#    Most-Recent has negative R^2 on PaO2 and Glucose -- this is real.)
# ---------------------------------------------------------------------------
data_regression = {
    'panel_label': 'Regression',
    'metric': r'$R^2$',
    'tasks': ['Creatinine', 'PaO2', 'Platelets', 'Glucose'],
    'always_show': ['Most-Recent', 'MOTOR'],
    'methods': [
        'Most-Recent',
        'CC-Llama',
        'CC-Mamba',
        'MOTOR',
        'Transformer-NTP',
        'Transformer-TPP',
        'Transformer-SurvivEHR',
        'Transformer-ICareFM',
        'Transformer-ORA',
        'Mamba-NTP',
        'Mamba-TPP',
        'Mamba-ORA',
    ],
    'mean': np.array([
        [0.877, -0.431, 0.910, -0.030],   # Most-Recent
        [0.319,  0.219, 0.290,  0.155],   # CC-Llama
        [0.307,  0.218, 0.261,  0.148],   # CC-Mamba
        [0.556,  0.231, 0.298,  0.163],   # MOTOR
        [0.489,  0.259, 0.312,  0.180],   # Transformer-NTP
        [0.603,  0.234, 0.310,  0.154],   # Transformer-TPP
        [0.607,  0.185, 0.437,  0.196],   # Transformer-SurvivEHR
        [0.784,  0.284, 0.844,  0.134],   # Transformer-ICareFM
        [0.603,  0.267, 0.605,  0.186],   # Transformer-ORA
        [0.595,  0.270, 0.407,  0.202],   # Mamba-NTP
        [0.617,  0.241, 0.400,  0.160],   # Mamba-TPP
        [0.645,  0.286, 0.659,  0.166],   # Mamba-ORA
    ]),
    'std': np.array([
        [0.034, 0.024, 0.002, 0.035],   # Most-Recent
        [0.018, 0.005, 0.004, 0.015],   # CC-Llama
        [0.017, 0.005, 0.004, 0.014],   # CC-Mamba
        [0.033, 0.005, 0.004, 0.016],   # MOTOR
        [0.025, 0.005, 0.004, 0.017],   # Transformer-NTP
        [0.030, 0.005, 0.003, 0.015],   # Transformer-TPP
        [0.033, 0.004, 0.004, 0.019],   # Transformer-SurvivEHR
        [0.044, 0.006, 0.002, 0.019],   # Transformer-ICareFM
        [0.028, 0.005, 0.003, 0.018],   # Transformer-ORA
        [0.029, 0.005, 0.003, 0.019],   # Mamba-NTP
        [0.030, 0.005, 0.004, 0.016],   # Mamba-TPP
        [0.034, 0.006, 0.003, 0.015],   # Mamba-ORA
    ]),
}


# Auto-attach colors to each panel after methods are defined.
for _d in (data_classification, data_tte, data_regression):
    _d['colors'] = [COLORS[m] for m in _d['methods']]
