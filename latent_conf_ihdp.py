#!/usr/bin/env python3
"""
latent_conf_ihdp.py

This script processes the IHDP concatenated CSV (with 25 covariates) to extract a latent confounder.
It performs the following:
  1. Identifies the ordinal covariate (among x1 ... x25) that is most correlated with the continuous outcome y_factual.
  2. One–hot encodes that covariate.
  3. Replicates the one–hot encoding three times and pads/trims to create a 30-dimensional vector.
  4. For each flipping probability p in {0.1, 0.2, 0.3, 0.4, 0.5}, flips each bit with probability p
     to create five noisy versions of the latent confounder Z.
  5. Removes the selected covariate from the original set, saving a modified covariate dataset with 24 columns.
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def is_ordinal(values, tol=1e-5):
    """Return True if all values are nearly integers."""
    return np.allclose(values, np.round(values), atol=tol)

def main():
    parser = argparse.ArgumentParser(
        description="Extract the most correlated ordinal covariate as a latent confounder Z and remove it from X."
    )
    parser.add_argument("--input-file", type=str, default="data/IHDP/csv/concatenated_ihdp.csv",
                        help="Path to the IHDP concatenated CSV file.")
    parser.add_argument("--output-x", type=str, default="data/IHDP/processed_X_ihdp_modified.csv",
                        help="Path to save modified IHDP covariates (24 columns).")
    parser.add_argument("--output-z-prefix", type=str, default="data/IHDP/processed_Z_ihdp",
                        help="Output prefix for latent confounder Z datasets. (Suffix will include flip probability.)")
    args = parser.parse_args()

    # Load the concatenated IHDP CSV.
    df = pd.read_csv(args.input_file)
    expected_cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1,26)]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # Outcome is y_factual.
    outcome = df["y_factual"].values.astype(float)

    # Identify ordinal covariates among x1...x25.
    ordinal_corr = {}
    for i in range(1,26):
        col = f"x{i}"
        values = df[col].values.astype(float)
        if is_ordinal(values):
            corr, _ = pearsonr(values, outcome)
            ordinal_corr[col] = abs(corr)
    if not ordinal_corr:
        raise ValueError("No ordinal covariate found among x1...x25.")
    # Select the ordinal covariate with highest absolute correlation.
    selected_cov = max(ordinal_corr, key=ordinal_corr.get)
    print("Selected ordinal covariate:", selected_cov, "with absolute correlation:", ordinal_corr[selected_cov])

    # One-hot encode the selected covariate.
    selected_values = df[selected_cov].values.astype(int)
    categories = np.unique(selected_values)
    one_hot = np.zeros((len(selected_values), len(categories)), dtype=int)
    for idx, cat in enumerate(categories):
        one_hot[:, idx] = (selected_values == cat).astype(int)
    
    # Replicate the one-hot encoding three times.
    replicated = np.tile(one_hot, (1, 3))
    d = replicated.shape[1]
    # Trim or pad to 30 dimensions.
    if d > 30:
        replicated = replicated[:, :30]
    elif d < 30:
        pad_width = 30 - d
        replicated = np.hstack([replicated, np.zeros((replicated.shape[0], pad_width), dtype=int)])
    
    # Generate five versions of Z by flipping bits with probability p.
    flip_ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    for p in flip_ps:
        noisy_Z = replicated.copy()
        flip_mask = np.random.rand(*noisy_Z.shape) < p
        noisy_Z[flip_mask] = 1 - noisy_Z[flip_mask]
        out_file = f"{args.output_z_prefix}_p{p}.csv"
        pd.DataFrame(noisy_Z, columns=[f"z{i+1}" for i in range(noisy_Z.shape[1])]).to_csv(out_file, index=False)
        print(f"Noisy latent confounder Z with p={p} saved to: {out_file}")

    # Remove the selected covariate from X.
    covariate_cols = [f"x{i}" for i in range(1,26)]
    remaining_covs = [col for col in covariate_cols if col != selected_cov]
    modified_X = df[remaining_covs]
    modified_X.to_csv(args.output_x, index=False)
    print(f"Modified IHDP covariate dataset (24 covariates) saved to: {args.output_x}")

if __name__ == "__main__":
    main()
