# src/drift_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency

def detect_drift(reference_csv, current_csv, alpha=0.05, show_plots=False):
    """
    Compare the distributions of features between two datasets to detect data drift.

    Parameters:
    - reference_csv (str): File path to the reference (baseline) dataset.
    - current_csv (str): File path to the current (incoming) dataset.
    - alpha (float): Significance level to determine drift (default: 0.05).
    - show_plots (bool): If True, plots the distribution comparison of features.

    Returns:
    - drift_results (list): A list of dictionaries containing feature drift information.
    """

    # Load both datasets
    reference_df = pd.read_csv(reference_csv)
    current_df = pd.read_csv(current_csv)

    drift_results = []  # Stores results for each feature

    for col in reference_df.columns:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            # Perform Kolmogorov–Smirnov test for numerical columns
            stat, p = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            drift = p < alpha

            result = {
                "feature": col,
                "type": "numerical",
                "test": "KS-test",
                "p_value": p,
                "drift": drift
            }

            # Optional: Plot KDE distribution of the numerical feature
            if show_plots:
                plt.figure(figsize=(6, 3))
                sns.kdeplot(reference_df[col], label="Reference", fill=True)
                sns.kdeplot(current_df[col], label="Current", fill=True)
                plt.title(f"Distribution: {col}")
                plt.legend()
                plt.tight_layout()
                plt.show()

        else:
            # Perform Chi-square test for categorical columns
            ref_counts = reference_df[col].value_counts()
            cur_counts = current_df[col].value_counts()

            # Align both series to contain all possible categories
            all_categories = list(set(ref_counts.index) | set(cur_counts.index))
            ref_counts = ref_counts.reindex(all_categories, fill_value=0)
            cur_counts = cur_counts.reindex(all_categories, fill_value=0)

            contingency = np.array([ref_counts, cur_counts])
            stat, p, _, _ = chi2_contingency(contingency)
            drift = p < alpha

            result = {
                "feature": col,
                "type": "categorical",
                "test": "Chi-square",
                "p_value": p,
                "drift": drift
            }

        # Append this feature's result to the overall list
        drift_results.append(result)

    return drift_results
