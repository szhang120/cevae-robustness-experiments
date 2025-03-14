# ======================
# File: data.py
# ======================

import torch
import pandas as pd
import numpy as np
from pathlib import Path

##########################
# Twin Data Functions
##########################

def load_twins_data(path_x: str = "data/TWINS/processed_X.csv",
                    path_t: str = "data/TWINS/processed_t.csv",
                    path_y: str = "data/TWINS/processed_y.csv",
                    path_z: str = "data/TWINS/processed_z_p0.1.csv") -> tuple:
    """
    Loads the Twins dataset. Returns X, t, y, Z as torch Tensors.
    """
    X = pd.read_csv(path_x).values
    t = pd.read_csv(path_t).values.squeeze()
    y = pd.read_csv(path_y).values.squeeze()
    Z = pd.read_csv(path_z).values

    X = torch.tensor(X, dtype=torch.float)
    t = torch.tensor(t, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    Z = torch.tensor(Z, dtype=torch.float)

    total_samples = X.shape[0]
    if total_samples % 2 != 0:
        raise ValueError("Number of samples must be even for Twins data.")

    return X, t, y, Z


def prepare_train_test_split(X: torch.Tensor, t: torch.Tensor, y: torch.Tensor, Z: torch.Tensor,
                             num_data: int, test_size: float, seed: int) -> tuple:
    """
    Given X, t, y, Z, creates a train/test split for the Twins dataset.
    Splitting is done at the twin-pair level to preserve pairing.
    Returns:
      - X_train, t_train, y_train, Z_train (training tensors)
      - X_test, t_test, y_test, Z_test (test tensors)
      - true_ite_train, true_ite_test (computed from observed outcomes)
      - XZ_train, t_train_np, y_train_np (for baseline methods)
      - train_twin0, train_twin1, test_twin0, test_twin1 (index arrays)
    """
    total_samples = X.shape[0]
    N_pairs = total_samples // 2
    sample_size_pairs = num_data // 2

    torch.manual_seed(seed)
    selected_pairs = torch.randperm(N_pairs)[:sample_size_pairs]

    n_test = int(test_size * sample_size_pairs)
    test_pairs = selected_pairs[:n_test]
    train_pairs = selected_pairs[n_test:]

    train_twin0 = 2 * train_pairs
    train_twin1 = 2 * train_pairs + 1
    test_twin0 = 2 * test_pairs
    test_twin1 = 2 * test_pairs + 1

    X_train = torch.cat([X[train_twin0], X[train_twin1]], dim=0)
    t_train = torch.cat([t[train_twin0], t[train_twin1]], dim=0)
    y_train = torch.cat([y[train_twin0], y[train_twin1]], dim=0)
    Z_train = torch.cat([Z[train_twin0], Z[train_twin1]], dim=0)

    X_test = torch.cat([X[test_twin0], X[test_twin1]], dim=0)
    t_test = torch.cat([t[test_twin0], t[test_twin1]], dim=0)
    y_test = torch.cat([y[test_twin0], y[test_twin1]], dim=0)
    Z_test = torch.cat([Z[test_twin0], Z[test_twin1]], dim=0)

    true_ite_train = (y[train_twin1] - y[train_twin0]).numpy()
    true_ite_test = (y[test_twin1] - y[test_twin0]).numpy()

    XZ_train = np.concatenate([X_train.numpy(), Z_train.numpy()], axis=1)
    t_train_np = t_train.numpy()
    y_train_np = y_train.numpy()

    return (X_train, t_train, y_train, Z_train,
            X_test, t_test, y_test, Z_test,
            true_ite_train, true_ite_test,
            XZ_train, t_train_np, y_train_np,
            train_twin0, train_twin1, test_twin0, test_twin1)

    
##########################
# IHDP Data Functions
##########################

def load_ihdp_data(path: str = "data/IHDP/csv/concatenated_ihdp.csv",
                   train_shifted: bool = False) -> tuple:
    """
    Loads the IHDP dataset from the concatenated CSV file.

    The expected columns are: treatment, y_factual, y_cfactual, mu0, mu1, x1...x25.
    (Both factual and counterfactual outcomes are available.)

    Depending on the 'train_shifted' flag, this function loads either the PCA‐shifted
    covariates (for training) or the modified (24-column) covariates (for testing).
    For the latent confounder Z, it attempts to load a default file (flip probability 0.1).

    Returns:
        X: Covariate tensor.
        t: Treatment tensor.
        y: Factual outcome tensor (y_factual).
        y_cf: Counterfactual outcome tensor (y_cfactual).
        Z: Latent confounder tensor.
    """
    df = pd.read_csv(path)
    expected_cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # Extract treatment and outcomes.
    t = df["treatment"].values
    y = df["y_factual"].values
    y_cf = df["y_cfactual"].values

    # Determine which covariate file to load
    x_mod_file = Path("data/IHDP/processed_X_ihdp_modified.csv")
    x_pca_file = Path("data/IHDP/processed_X_pca_shifted.csv")

    if train_shifted:
        if not x_pca_file.exists():
            raise FileNotFoundError(f"--train-shifted=True but '{x_pca_file}' not found.")
        X = pd.read_csv(x_pca_file).values
        print("Loaded PCA-shifted IHDP covariates from:", x_pca_file)
    else:
        if x_mod_file.exists():
            X = pd.read_csv(x_mod_file).values
            print("Loaded modified IHDP covariates (24 columns) from:", x_mod_file)
        else:
            # Fallback to original 25 covariates
            covariate_cols = [f"x{i}" for i in range(1, 26)]
            X = df[covariate_cols].values
            print("Using original 25-column IHDP covariates from the concatenated CSV.")

    # Load latent confounder Z (default flip probability 0.1)
    z_file = Path("data/IHDP/processed_Z_ihdp_p0.1.csv")
    if z_file.exists():
        Z = pd.read_csv(z_file).values
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        print("Loaded latent confounder Z from:", z_file)
    else:
        Z = np.zeros((X.shape[0], 1))
        print("No latent confounder file found; using Z=0.")

    # Convert arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float)
    t = torch.tensor(t, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    y_cf = torch.tensor(y_cf, dtype=torch.float)
    Z = torch.tensor(Z, dtype=torch.float)

    return X, t, y, y_cf, Z


def prepare_train_test_split_ihdp(X: torch.Tensor, t: torch.Tensor, y: torch.Tensor, y_cf: torch.Tensor, Z: torch.Tensor,
                                  test_size: float, seed: int) -> tuple:
    """
    Prepares a random train/test split for IHDP.

    The training set uses the (possibly shifted) covariates as loaded.
    The test set is forced to come from the original unshifted modified covariates.

    This function now computes the true ITE from the available factual and counterfactual outcomes.
    True ITE is computed as:
        if t == 1:  y_factual - y_cfactual
        else:       y_cfactual - y_factual

    Returns a tuple containing:
      - X_train, t_train, y_train, y_cf_train, Z_train (training tensors)
      - X_test, t_test, y_test, y_cf_test, Z_test (test tensors; note X_test is unshifted)
      - true_ite_train, true_ite_test (computed true ITE arrays)
      - XZ_train, t_train_np, y_train_np (for baseline methods)
      - train_idx, test_idx: index arrays for training and testing splits.
    """
    np.random.seed(seed)
    total_samples = X.shape[0]
    indices = np.random.permutation(total_samples)
    n_test = int(test_size * total_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    # Training set: use the current (possibly shifted) data
    X_train = X[train_idx]
    t_train = t[train_idx]
    y_train = y[train_idx]
    y_cf_train = y_cf[train_idx]
    Z_train = Z[train_idx]

    # Test set: load unshifted covariates from the modified file
    x_unshifted_path = Path("data/IHDP/processed_X_ihdp_modified.csv")
    if not x_unshifted_path.exists():
        raise FileNotFoundError(f"Can't locate unshifted test file: {x_unshifted_path}")
    X_unshifted = pd.read_csv(x_unshifted_path).values
    X_unshifted = torch.tensor(X_unshifted, dtype=torch.float)

    X_test = X_unshifted[test_idx]
    t_test = t[test_idx]
    y_test = y[test_idx]
    y_cf_test = y_cf[test_idx]
    Z_test = Z[test_idx]

    # Compute true ITE: if t==1 then y - y_cf else y_cf - y
    true_ite_train = np.where(t_train.numpy()==1, (y_train - y_cf_train).numpy(), (y_cf_train - y_train).numpy())
    true_ite_test  = np.where(t_test.numpy()==1, (y_test - y_cf_test).numpy(), (y_cf_test - y_test).numpy())

    # For baseline methods, combine X and Z for training
    XZ_train = np.concatenate([X_train.numpy(), Z_train.numpy()], axis=1)
    t_train_np = t_train.numpy()
    y_train_np = y_train.numpy()

    return (X_train, t_train, y_train, y_cf_train, Z_train,
            X_test, t_test, y_test, y_cf_test, Z_test,
            true_ite_train, true_ite_test,
            XZ_train, t_train_np, y_train_np,
            train_idx, test_idx)

    
if __name__ == "__main__":
    print("Module data.py loaded successfully.")
