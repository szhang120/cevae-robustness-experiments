#!/usr/bin/env python3
"""
pca_shift_ihdp_quantile.py

This script performs PCA on the modified IHDP covariates (24 covariates)
and applies a nonlinear, quantile-based shift on the top components.
It then reconstructs the shifted covariates for training.

Rationale:
  - A purely linear flip or scale is largely undone by standard scaling.
  - This quantile-based shift warps the distribution in a continuous, nonlinear way,
    so that the transformation remains detectable even after inverse whitening.
  - The new transformation uses a piecewise power function that “bends” the
    empirical CDF more extremely (with gamma > 1, the mid-range values are pushed
    further away from the center), making the covariate shift even more pronounced.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extreme_quantile_shift(pc_values, gamma=2.0):
    """
    Apply an extreme nonlinear quantile-based shift to the PC values.
    
    For each value in pc_values, this function:
      1. Computes the empirical CDF (quantile) and rescales it to [0, 1].
      2. Applies a piecewise power transformation:
           - For q < 0.5:  f(q) = q^(1/gamma)
           - For q >= 0.5: f(q) = 1 - (1-q)^(1/gamma)
         This pushes mid-range values farther away from 0.5.
      3. Computes the shift as (f(q) - q)*data_range and adds it to the original value.
      4. Optionally adds small random noise to break ties.
    
    Parameters:
      pc_values : np.array, the PC scores for one dimension.
      gamma     : float, controls the magnitude of the nonlinearity. 
                       When gamma=1, f(q)=q (i.e. no shift). For gamma>1, the transformation
                       becomes more extreme.
                       
    Returns:
      np.array: The shifted PC values.
    """
    arr = pc_values.copy()
    n = len(arr)
    # Compute empirical CDF: rank each value uniformly between 0 and 1.
    sorted_indices = np.argsort(arr)
    ranks = np.empty_like(sorted_indices, dtype=float)
    ranks[sorted_indices] = np.linspace(0, 1, num=n)
    
    data_range = arr.max() - arr.min()
    
    # Apply piecewise power transformation on the quantiles:
    f = np.empty_like(ranks)
    lower_mask = ranks < 0.5
    upper_mask = ~lower_mask
    # For values below the median:
    f[lower_mask] = np.power(ranks[lower_mask], 1.0/gamma)
    # For values at or above the median:
    f[upper_mask] = 1 - np.power(1 - ranks[upper_mask], 1.0/gamma)
    
    # Compute shift: the difference between the transformed and original quantiles
    shift_amount = (f - ranks) * data_range
    
    # Add shift and optional small noise
    arr_shifted = arr + shift_amount
    noise_scale = 0.05 * (data_range + 1e-8)
    arr_shifted += np.random.normal(loc=0, scale=noise_scale, size=arr.shape)
    
    return arr_shifted

def plot_pc_distribution(before, after, pc_index, plot_file):
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
    print(f"Plot saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Perform PCA on modified IHDP covariates (24 columns) and apply an extreme nonlinear quantile shift on the top PCs."
    )
    parser.add_argument("--input-file", type=str, default="data/IHDP/processed_X_ihdp_modified.csv",
                        help="Modified IHDP covariate file (24 covariates).")
    parser.add_argument("--output-file", type=str, default="data/IHDP/processed_X_pca_shifted.csv",
                        help="Output file for the PCA-shifted IHDP covariates.")
    parser.add_argument("--n-components", type=int, default=24,
                        help="Number of PCA components to compute (default: 24)")
    parser.add_argument("--plot-dir", type=str, default="pca_shift_plots",
                        help="Directory to save the PC distribution plots.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Tuning parameter for the magnitude of the extreme quantile shift (default: 2.0)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        return

    df = pd.read_csv(args.input_file)
    covariate_cols = df.columns.tolist()  # Expecting 24 covariates.
    X = df.values  # shape: (n_samples, 24)

    # Standardize the covariates.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Perform PCA.
    pca = PCA(n_components=args.n_components)
    PCs = pca.fit_transform(X_std)

    print("Explained Variance Ratios:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        suffix = " (top PCs)" if i < 5 else ""
        print(f"  PC{i+1}{suffix}: {ratio:.2%}")

    # Copy PCs to be shifted.
    PCs_shifted = PCs.copy()
    # We apply the extreme shift on the top 5 components.
    top_k = 24
    if PCs.shape[1] < top_k:
        raise ValueError(f"Found {PCs.shape[1]} components, need at least {top_k}.")

    for i in range(top_k):
        original_pc = PCs[:, i]
        shifted_pc = extreme_quantile_shift(original_pc, gamma=args.gamma)
        PCs_shifted[:, i] = shifted_pc

    # Plot distributions for the top 5 PCs.
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)
    for i in range(top_k):
        plot_file = os.path.join(args.plot_dir, f"PC{i+1}_before_after.png")
        plot_pc_distribution(PCs[:, i], PCs_shifted[:, i], i, plot_file)

    # Inverse transform back to the original feature space.
    X_shifted_std = pca.inverse_transform(PCs_shifted)
    X_shifted = scaler.inverse_transform(X_shifted_std)

    df_shifted = pd.DataFrame(X_shifted, columns=covariate_cols)
    df_shifted.to_csv(args.output_file, index=False)
    print(f"\nPCA-shifted IHDP covariate dataset saved to: {args.output_file}")
    print("Top 5 PCs were transformed via an extreme nonlinear quantile-based shift to induce a robust covariate shift.")

if __name__ == "__main__":
    main()
