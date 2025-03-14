#!/usr/bin/env python3
"""
contrast_ihdp.py

This script contrasts two IHDP covariate datasets:
  - The original modified IHDP covariate dataset (24 covariates), e.g.:
      data/IHDP/processed_X_ihdp_modified.csv
  - The PCA-shifted IHDP covariate dataset, e.g.:
      data/IHDP/processed_X_pca_shifted.csv

For each column, the script computes summary statistics (mean and standard deviation)
and performs a Kolmogorov–Smirnov (KS) test to statistically compare the distributions.
Overlay histograms for each column are saved to a specified output directory.
A summary table is printed at the end.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def compare_column(col_orig, col_shifted, col_name, plot_dir):
    """
    Compare a column from the original and shifted datasets.
    Computes mean, standard deviation, and KS test (statistic and p-value).
    Also creates an overlay histogram plot saved in plot_dir.
    """
    mean_orig = col_orig.mean()
    std_orig = col_orig.std()
    mean_shift = col_shifted.mean()
    std_shift = col_shifted.std()
    
    # Perform KS test on non-NaN values.
    ks_stat, ks_pvalue = ks_2samp(col_orig.dropna(), col_shifted.dropna())
    
    # Plot overlay histograms.
    plt.figure(figsize=(8, 6))
    plt.hist(col_orig.dropna(), bins=50, alpha=0.5, density=True, label="Original")
    plt.hist(col_shifted.dropna(), bins=50, alpha=0.5, density=True, label="Shifted")
    plt.title(f"Distribution Comparison for '{col_name}'")
    plt.xlabel(col_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(plot_dir, f"{col_name}_contrast.png")
    plt.savefig(plot_file)
    plt.close()
    
    return {
        "mean_orig": mean_orig,
        "std_orig": std_orig,
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "plot_file": plot_file
    }

def main():
    parser = argparse.ArgumentParser(
        description="Contrast IHDP covariate datasets (original modified vs. PCA shifted) and assess distribution differences."
    )
    parser.add_argument("--original", type=str, default="data/IHDP/processed_X_ihdp_modified.csv",
                        help="Path to the original modified IHDP covariate dataset (default: data/IHDP/processed_X_ihdp_modified.csv)")
    parser.add_argument("--shifted", type=str, default="data/IHDP/processed_X_pca_shifted.csv",
                        help="Path to the PCA shifted IHDP covariate dataset (default: data/IHDP/processed_X_pca_shifted.csv)")
    parser.add_argument("--plot-dir", type=str, default="contrast_ihdp_plots",
                        help="Directory to save overlay plots (default: contrast_ihdp_plots)")
    args = parser.parse_args()
    
    # Check that both files exist.
    if not os.path.exists(args.original):
        print(f"Error: Original file '{args.original}' not found.")
        return
    if not os.path.exists(args.shifted):
        print(f"Error: Shifted file '{args.shifted}' not found.")
        return
    
    # Create the plot directory if it does not exist.
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load the two datasets.
    df_orig = pd.read_csv(args.original)
    df_shift = pd.read_csv(args.shifted)
    
    # Use common columns in case the headers differ.
    common_cols = list(set(df_orig.columns) & set(df_shift.columns))
    common_cols.sort()  # sort alphabetically for consistency
    if len(common_cols) == 0:
        print("Error: No common columns found between the two datasets.")
        return
    df_orig = df_orig[common_cols]
    df_shift = df_shift[common_cols]
    
    print("Comparing the following columns:")
    for col in common_cols:
        print(f"  {col}")
    
    # Compare each column.
    results = {}
    for col in common_cols:
        stats = compare_column(df_orig[col], df_shift[col], col, args.plot_dir)
        results[col] = stats
        print(f"Column: {col}")
        print(f"  Original: Mean = {stats['mean_orig']:.4f}, Std = {stats['std_orig']:.4f}")
        print(f"  Shifted:  Mean = {stats['mean_shift']:.4f}, Std = {stats['std_shift']:.4f}")
        print(f"  KS Statistic = {stats['ks_stat']:.4f}, p-value = {stats['ks_pvalue']:.4e}")
        print(f"  Plot saved to: {stats['plot_file']}\n")
    
    # Create a summary DataFrame.
    summary_rows = []
    for col, stat in results.items():
        summary_rows.append({
            "Column": col,
            "Mean (Original)": stat["mean_orig"],
            "Std (Original)": stat["std_orig"],
            "Mean (Shifted)": stat["mean_shift"],
            "Std (Shifted)": stat["std_shift"],
            "KS Statistic": stat["ks_stat"],
            "KS p-value": stat["ks_pvalue"]
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary of IHDP Covariate Comparisons:")
    print(summary_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()
