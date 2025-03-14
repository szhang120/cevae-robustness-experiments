#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Paths to the data files.
    shifted_path = "data/IHDP/processed_X_pca_shifted.csv"
    original_path = "data/IHDP/processed_X_ihdp_modified.csv"

    # Check if files exist.
    if not os.path.exists(shifted_path) or not os.path.exists(original_path):
        print("Required data files not found. Please ensure that both files exist in 'data/TWINS/'.")
        return

    # Load data as DataFrames.
    df_shifted = pd.read_csv(shifted_path)
    df_original = pd.read_csv(original_path)

    # Check that the data have the same columns.
    if not all(df_original.columns == df_shifted.columns):
        print("Column mismatch between original and shifted data.")
        return

    columns = df_original.columns
    shifted_vars = []
    diff_summary = []

    # Create a directory to save the distribution plots.
    plots_dir = "shift_analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Loop over each variable (column) and compare distributions.
    for col in columns:
        orig = df_original[col]
        shft = df_shifted[col]

        # Calculate summary statistics.
        orig_mean = orig.mean()
        shft_mean = shft.mean()
        mean_diff = shft_mean - orig_mean
        orig_std = orig.std()
        shft_std = shft.std()
        std_diff = shft_std - orig_std

        # Determine if the variable is shifted.
        # Here we use a threshold: if the absolute mean difference is larger than 10% of the original standard deviation.
        threshold = 0.1 * orig_std if orig_std != 0 else 0
        if abs(mean_diff) > threshold:
            shifted_vars.append(col)

        diff_summary.append({
            "column": col,
            "original_mean": orig_mean,
            "shifted_mean": shft_mean,
            "mean_diff": mean_diff,
            "original_std": orig_std,
            "shifted_std": shft_std,
            "std_diff": std_diff
        })

        # Plot histogram overlay for the variable.
        plt.figure(figsize=(8, 4))
        plt.hist(orig, bins=50, alpha=0.5, density=True, label="Original")
        plt.hist(shft, bins=50, alpha=0.5, density=True, label="Shifted")
        plt.title(f"Distribution Comparison for {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"distribution_{col}.png"))
        plt.close()

    # Print summary.
    print("Variables detected as shifted:")
    if shifted_vars:
        for var in shifted_vars:
            print(f"  {var}")
    else:
        print("  None")

    print("\nDetailed Summary (per variable):")
    for entry in diff_summary:
        print(f"{entry['column']}: original_mean={entry['original_mean']:.4f}, shifted_mean={entry['shifted_mean']:.4f}, "
              f"mean_diff={entry['mean_diff']:.4f}, original_std={entry['original_std']:.4f}, shifted_std={entry['shifted_std']:.4f}, "
              f"std_diff={entry['std_diff']:.4f}")

if __name__ == "__main__":
    main()
