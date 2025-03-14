#!/usr/bin/env python3
"""
pca_shift_ihdp.py

This script performs PCA on the IHDP covariates (columns x1 to x25 from the concatenated IHDP CSV),
then (a) induces a maximal covariate shift on the largest principal component (PC1) by flipping its values,
and (b) extracts a “latent confounder” from a smaller (but not the smallest) principal component (PC2).

The outputs are:
  - A new covariate dataset with the maximal shift applied (saved as processed_X_pca_shifted_max.csv),
    which retains the same header (x1,...,x25) as the original IHDP covariate file.
  - A latent confounder dataset (a single-column CSV, saved as processed_Z_pca.csv) with the PC2 values.

These files are created in the data/IHDP/ directory so that the IHDP data‐loader can use them for training/evaluation.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def flip_values(arr):
    """
    Given a 1D numpy array, return a flipped version:
    new_value = max + min - original.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return max_val + min_val - arr

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
    print(f"Plot saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Perform PCA on IHDP covariates, induce maximal shift on PC1, and extract PC2 as latent confounder Z."
    )
    parser.add_argument("--input-file", type=str, default="data/IHDP/csv/concatenated_ihdp.csv",
                        help="Path to the IHDP concatenated CSV file (default: data/IHDP/csv/concatenated_ihdp.csv)")
    parser.add_argument("--n-components", type=int, default=25,
                        help="Number of PCA components to compute (default: 25, equal to number of covariates)")
    parser.add_argument("--plot-dir", type=str, default="pca_shift_plots",
                        help="Directory to save the PC before/after plots (default: pca_shift_plots)")
    args = parser.parse_args()

    # Load the IHDP concatenated CSV.
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return

    df_full = pd.read_csv(args.input_file)
    # Expect these columns:
    expected_cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1,26)]
    if not all(col in df_full.columns for col in expected_cols):
        raise ValueError("The IHDP CSV file does not contain the expected columns.")

    # Extract covariates (assumed to be x1,...,x25)
    covariate_cols = [f"x{i}" for i in range(1,26)]
    X_orig = df_full[covariate_cols].values  # shape (n_samples, 25)

    # Standardize the covariates.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_orig)

    # Perform PCA.
    pca = PCA(n_components=args.n_components)
    PCs = pca.fit_transform(X_std)
    print("Explained variance ratios for computed components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")

    # ---- Dataset 1: Maximal shift on the largest PC (PC1) ----
    # Copy the PCA scores and flip the first principal component.
    PCs_shifted = np.copy(PCs)
    original_pc1 = PCs[:, 0]
    flipped_pc1 = flip_values(original_pc1)
    PCs_shifted[:, 0] = flipped_pc1

    # (Optional) Plot PC1 distribution before and after shifting.
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)
    plot_file_pc1 = os.path.join(args.plot_dir, "PC1_before_after.png")
    plot_pc_distribution(original_pc1, flipped_pc1, 0, plot_file_pc1)

    # Reconstruct the shifted covariates.
    X_shifted_std = pca.inverse_transform(PCs_shifted)
    X_shifted = scaler.inverse_transform(X_shifted_std)
    # Create a DataFrame with the same covariate column names.
    df_X_shifted = pd.DataFrame(X_shifted, columns=covariate_cols)
    # Save the new covariate dataset.
    output_X_file = os.path.join("data", "IHDP", "processed_X_pca_shifted_max.csv")
    df_X_shifted.to_csv(output_X_file, index=False)
    print(f"New IHDP covariate dataset with maximal shift on PC1 saved to: {output_X_file}")

    # ---- Dataset 2: Encode a smaller-but-not-the-smallest PC (choose PC2) as latent confounder Z ----
    # Here we take PC2 (index 1) from the original PCA (from standardized data).
    latent_conf = PCs[:, 1]  # shape (n_samples,)
    # Optionally, plot PC2 distribution.
    plot_file_pc2 = os.path.join(args.plot_dir, "PC2_distribution.png")
    plt.figure(figsize=(8,6))
    plt.hist(latent_conf, bins=50, alpha=0.7, density=True)
    plt.title("PC2 Distribution (Used as Latent Confounder Z)")
    plt.xlabel("PC2 Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(plot_file_pc2)
    plt.close()
    print(f"PC2 distribution plot saved to: {plot_file_pc2}")

    # Save the latent confounder as a one-column CSV.
    df_Z = pd.DataFrame(latent_conf, columns=["latent_conf"])
    output_Z_file = os.path.join("data", "IHDP", "processed_Z_pca.csv")
    df_Z.to_csv(output_Z_file, index=False)
    print(f"Latent confounder Z (from PC2) saved to: {output_Z_file}")

if __name__ == "__main__":
    main()
