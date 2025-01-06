import torch
import pandas as pd
import numpy as np

# replace with your own data paths and modify p = 0.0, 0.1, ..., 0.5 accordingly. 
def load_twins_data(path_x="data/TWINS/processed_X.csv",
                    path_t="data/TWINS/processed_t.csv",
                    path_y="data/TWINS/processed_y.csv",
                    path_z="data/TWINS/processed_z_p0.1.csv"):
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
        raise ValueError("Number of samples must be even.")

    return X, t, y, Z


def prepare_train_test_split(X, t, y, Z, num_data, test_size, seed):
    """
    Given X, t, y, Z, create a train/test split based on the argument parameters.
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

    # train/test
    X_train = torch.cat([X[train_twin0], X[train_twin1]], dim=0)
    t_train = torch.cat([t[train_twin0], t[train_twin1]], dim=0)
    y_train = torch.cat([y[train_twin0], y[train_twin1]], dim=0)
    Z_train = torch.cat([Z[train_twin0], Z[train_twin1]], dim=0)

    X_test = torch.cat([X[test_twin0], X[test_twin1]], dim=0)
    t_test = torch.cat([t[test_twin0], t[test_twin1]], dim=0)
    y_test = torch.cat([y[test_twin0], y[test_twin1]], dim=0)
    Z_test = torch.cat([Z[test_twin0], Z[test_twin1]], dim=0)

    # true ITE for train and test
    true_ite_train = (y[train_twin1] - y[train_twin0]).numpy()
    true_ite_test = (y[test_twin1] - y[test_twin0]).numpy()

    # combined XZ for "noncausal" methods
    XZ_train = np.concatenate([X_train.numpy(), Z_train.numpy()], axis=1)
    t_train_np = t_train.numpy()
    y_train_np = y_train.numpy()

    return (X_train, t_train, y_train, Z_train,
            X_test,  t_test,  y_test,  Z_test,
            true_ite_train, true_ite_test,
            XZ_train, t_train_np, y_train_np,
            train_twin0, train_twin1, test_twin0, test_twin1)
