#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def flip_column(series):
    """
    If the series is binary (contains only 0 and 1 or their string equivalents),
    flip its values (0->1, 1->0). Otherwise, perform a min-max flip:
    new_value = max + min - original.
    """
    unique_vals = series.dropna().unique()
    # Check if binary: allow values 0 and 1 (or strings '0' and '1')
    if set(unique_vals).issubset({0, 1, '0', '1'}):
        numeric_series = pd.to_numeric(series, errors='coerce')
        return 1 - numeric_series
    else:
        min_val = series.min()
        max_val = series.max()
        return max_val + min_val - series

def plot_distributions(before, after, column, plot_file):
    """
    Plot overlay histograms for the given column showing before and after distributions.
    Saves the plot to plot_file.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(before.dropna(), bins=50, alpha=0.5, density=True, label="Before Shift")
    plt.hist(after.dropna(), bins=50, alpha=0.5, density=True, label="After Shift")
    plt.title(f"Distribution of '{column}' Before vs. After Shift")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply a shift to ALL covariates by flipping each column to its maximum."
    )
    parser.add_argument("--shifted-file", type=str, default="data/TWINS/processed_X_covariate_shifted.csv",
                        help="Path to the currently shifted dataset CSV file. (Default: data/TWINS/processed_X_covariate_shifted.csv)")
    parser.add_argument("--output-file", type=str, default="data/TWINS/processed_X_allshifted.csv",
                        help="Path to save the modified (all columns shifted) dataset. (Default: data/TWINS/processed_X_allshifted.csv)")
    parser.add_argument("--plot-dir", type=str, default="all_shift_plots",
                        help="Directory to save the before/after plots for each column. (Default: all_shift_plots)")
    args = parser.parse_args()

    # Check that the input file exists.
    if not os.path.exists(args.shifted_file):
        print(f"Error: Shifted file '{args.shifted_file}' does not exist.")
        return

    # Load the currently shifted dataset.
    df_shifted = pd.read_csv(args.shifted_file)
    
    # Create a copy of the dataset to apply the new shift.
    df_new = df_shifted.copy()

    # Ensure the plot directory exists.
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)

    # Iterate over every column in the dataset.
    for col in df_new.columns:
        before_shift = df_new[col].copy()
        # Apply the new shift (flip the values to their max)
        df_new[col] = flip_column(df_new[col])
        after_shift = df_new[col]
        
        # Generate and save the plot.
        plot_file = os.path.join(args.plot_dir, f"before_after_{col}.png")
        plot_distributions(before_shift, after_shift, col, plot_file)
        print(f"Column '{col}' has been shifted.")

    # Save the newly shifted dataset.
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df_new.to_csv(args.output_file, index=False)
    print(f"\nModified dataset with all covariates shifted saved to {args.output_file}")

if __name__ == "__main__":
    main()
