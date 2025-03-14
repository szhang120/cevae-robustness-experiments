#!/usr/bin/env python3
"""
shift_pca.py

This script performs PCA on the Twins covariates and then shifts (flips and scales)
the top 3 principal components. The transformation used is:
    new_value = scale_factor * (max + min - original)
for the top 3 components, leaving the remaining components unchanged.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def flip_values(arr, scale_factor=2.0):
    """
    Given a 1D numpy array, return a flipped and scaled version:
    new_value = scale_factor * (max + min - original).
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return scale_factor * (max_val + min_val - arr)

def plot_pc_distributions(before, after, pc_index, plot_file):
    """
    Plot overlay histograms for a given principal component (pc_index)
    before and after shifting.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(before, bins=50, alpha=0.5, density=True, label="Before Shift")
    plt.hist(after, bins=50, alpha=0.5, density=True, label="After Shift")
    plt.title(f"PC{pc_index+1} Distribution Before vs. After Shift")
    plt.xlabel(f"PC{pc_index+1} Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Perform PCA on Twins covariates and then shift (flip and scale) the top 3 principal components."
    )
    parser.add_argument("--input-file", type=str, default="data/TWINS/processed_X.csv",
                        help="Path to the input covariate CSV file (default: data/TWINS/processed_X.csv)")
    parser.add_argument("--output-file", type=str, default="data/TWINS/processed_X_pca_shifted.csv",
                        help="Path to save the new shifted covariate dataset (default: data/TWINS/processed_X_pca_shifted.csv)")
    parser.add_argument("--n-components", type=int, default=10,
                        help="Number of PCA components to compute (default: 10)")
    parser.add_argument("--plot-dir", type=str, default="pca_shift_plots",
                        help="Directory to save the before/after PC plots (default: pca_shift_plots)")
    parser.add_argument("--scale-factor", type=float, default=2.0,
                        help="Scaling factor to modify the flipped PCs (default: 2.0)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return

    df = pd.read_csv(args.input_file)
    print(f"Loaded data with shape {df.shape} from {args.input_file}")

    scaler = StandardScaler()
    X_std = scaler.fit_transform(df.values)

    pca = PCA(n_components=args.n_components)
    PCs = pca.fit_transform(X_std)
    print("Explained variance ratios for computed components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")

    top3_before = PCs[:, :3].copy()

    # Flip and scale the top 3 principal components.
    for i in range(3):
        PCs[:, i] = flip_values(PCs[:, i], scale_factor=args.scale_factor)
    
    top3_after = PCs[:, :3]

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)
    for i in range(3):
        plot_file = os.path.join(args.plot_dir, f"PC{i+1}_before_after.png")
        plot_pc_distributions(top3_before[:, i], top3_after[:, i], i, plot_file)

    X_new_std = pca.inverse_transform(PCs)
    X_new = scaler.inverse_transform(X_new_std)

    df_new = pd.DataFrame(X_new, columns=df.columns)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df_new.to_csv(args.output_file, index=False)
    print(f"New shifted dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
