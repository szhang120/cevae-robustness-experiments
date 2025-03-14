# ======================
# File: data.py
# ======================

import torch
import pandas as pd
import numpy as np

# DATALOADING: SET (1) SHIFT AND (2) P_BITFLIPPING.
def load_twins_data(path_x="data/TWINS/processed_X_covariate_shifted.csv",  # SHIFT by default
                    path_t="data/TWINS/processed_t.csv",
                    path_y="data/TWINS/processed_y.csv",
                    path_z="data/TWINS/processed_z_p0.5.csv"):
    """
    Loads the Twins dataset. Returns X, t, y, Z as torch Tensors.
    By default, uses the shifted X for training. (If you want unshifted for training,
    change path_x to "data/TWINS/processed_X.csv".)
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
        raise ValueError("Number of samples must be even.")

    return X, t, y, Z


def prepare_train_test_split(X, t, y, Z, num_data, test_size, seed):
    """
    Given X, t, y, Z, create a train/test split based on the argument parameters.
    IMPORTANT CHANGE: Even if 'X' is shifted (or not) for training, we now ALWAYS test on the unshifted X.
    """
    total_samples = X.shape[0]
    # Each 'pair' is (twin0, twin1).
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

    # ---------------------------
    # Train set uses whichever X was passed in (shifted or unshifted).
    # ---------------------------
    X_train = torch.cat([X[train_twin0], X[train_twin1]], dim=0)
    t_train = torch.cat([t[train_twin0], t[train_twin1]], dim=0)
    y_train = torch.cat([y[train_twin0], y[train_twin1]], dim=0)
    Z_train = torch.cat([Z[train_twin0], Z[train_twin1]], dim=0)

    # ---------------------------
    # ALWAYS load and use the unshifted X for the test set:
    # ---------------------------
    X_unshifted_all = pd.read_csv("data/TWINS/processed_X.csv").values
    X_unshifted_all = torch.tensor(X_unshifted_all, dtype=torch.float)

    X_test = torch.cat([X_unshifted_all[test_twin0], X_unshifted_all[test_twin1]], dim=0)
    t_test = torch.cat([t[test_twin0], t[test_twin1]], dim=0)
    y_test = torch.cat([y[test_twin0], y[test_twin1]], dim=0)
    Z_test = torch.cat([Z[test_twin0], Z[test_twin1]], dim=0)

    # True ITE for train and test (still uses the same Y for twin0 vs twin1)
    true_ite_train = (y[train_twin1] - y[train_twin0]).numpy()
    true_ite_test = (y[test_twin1] - y[test_twin0]).numpy()

    # Combined XZ for "noncausal" methods (training portion)
    XZ_train = np.concatenate([X_train.numpy(), Z_train.numpy()], axis=1)
    t_train_np = t_train.numpy()
    y_train_np = y_train.numpy()

    return (
        X_train, t_train, y_train, Z_train,
        X_test,  t_test,  y_test,  Z_test,
        true_ite_train, true_ite_test,
        XZ_train, t_train_np, y_train_np,
        train_twin0, train_twin1,
        test_twin0, test_twin1
    )
