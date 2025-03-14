# ======================
# File: main.py
# ======================

#!/usr/bin/env python3
import argparse
import logging
import torch
import pandas as pd
import numpy as np

# data-loading
from data import load_ihdp_data, prepare_train_test_split_ihdp
from data import load_twins_data, prepare_train_test_split

# training/evaluation
from train_test import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(
        description="CEVAE with Z and other model comparisons using combined XZ features."
    )
    parser.add_argument("--dataset", choices=["twins", "ihdp"], default="ihdp",
                        help="Dataset to use: twins or ihdp.")
    parser.add_argument("--num-data", default=23968, type=int,
                        help="Only used for twins dataset.")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--cv-folds", default=3, type=int)

    # Control whether we use the PCA-shifted file for IHDP
    parser.add_argument("--train-shifted", dest="train_shifted", action="store_true",
                        help="Use PCA-shifted covariates for IHDP.")
    parser.add_argument("--no-train-shifted", dest="train_shifted", action="store_false",
                        help="Use unshifted covariates for IHDP.")
    
    parser.set_defaults(train_shifted=True)  # <--- default is True

    # parser.add_argument("--feature-dim", default=24, type=int)
    # parser.add_argument("-n", "--num-epochs", default=20, type=int)
    # parser.add_argument("--latent-dim", default=30, type=int)
    # parser.add_argument("--hidden-dim", default=300, type=int)
    # parser.add_argument("--num-layers", default=2, type=int)
    # parser.add_argument("-b", "--batch-size", default=512, type=int)
    # parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    # parser.add_argument("-lrd", "--learning-rate-decay", default=0.95, type=float)
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    # parser.add_argument("--seed", default=23, type=int)
    # parser.add_argument("--jit", action="store_true")
    # parser.add_argument("--cuda", action="store_true")
    # parser.add_argument("--only-cevae", action="store_true", default=True,
    #                     help="If set, only train/evaluate CEVAE, skipping other models.")

     # NEW HYPERPARAMETER SETTINGS:
    parser.add_argument("--feature-dim", default=24, type=int)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)         # increased from 20 to 100 epochs
    parser.add_argument("--latent-dim", default=30, type=int)
    parser.add_argument("--hidden-dim", default=200, type=int)               # reduced hidden dimension from 300 to 200
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("-b", "--batch-size", default=512, type=int)
    parser.add_argument("-lr", "--learning-rate", default=5e-3, type=float)     # reduced learning rate from 1e-2 to 5e-3
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.98, type=float)  # slower decay
    parser.add_argument("--weight-decay", default=1e-6, type=float)            # reduced weight decay from 1e-5 to 1e-6
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--only-cevae", action="store_true", default=True,
                        help="If set, only train/evaluate CEVAE, skipping other models.")

    args = parser.parse_args()

    logging.getLogger("pyro").setLevel(logging.DEBUG)
    if logging.getLogger("pyro").handlers:
        logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

    if args.dataset == "ihdp":
        # -----------------------------------------
        # IHDP case
        # -----------------------------------------
        # Load IHDP data (including counterfactual outcomes)
        X_input, t, y, y_cf, Z = load_ihdp_data(train_shifted=args.train_shifted)
        
        # Split into train/test (the test set uses unshifted covariates)
        (X_train, t_train, y_train, y_cf_train, Z_train,
         X_test, t_test, y_test, y_cf_test, Z_test,
         true_ite_train, true_ite_test,
         XZ_train, t_train_np, y_train_np,
         train_idx, test_idx) = prepare_train_test_split_ihdp(
            X_input, t, y, y_cf, Z,
            test_size=args.test_size,
            seed=args.seed
        )

        dataset_flag = "ihdp"
        X_all = X_input  # for the function signature in train_and_evaluate
        extra_args = {}  # IHDP does not require twin-pair indices

    else:
        # -----------------------------------------
        # TWINS case
        # -----------------------------------------
        X, t, y, Z = load_twins_data()
        (X_train, t_train, y_train, Z_train,
         X_test, t_test, y_test, Z_test,
         true_ite_train, true_ite_test,
         XZ_train, t_train_np, y_train_np,
         train_twin0, train_twin1,
         test_twin0, test_twin1) = prepare_train_test_split(
            X, t, y, Z,
            num_data=args.num_data,
            test_size=args.test_size,
            seed=args.seed
        )
        dataset_flag = "twins"
        X_all = X  # for the function signature
        extra_args = {
            "train_twin0": train_twin0,
            "train_twin1": train_twin1,
            "test_twin0": test_twin0,
            "test_twin1": test_twin1
        }

    # Now call the unified train_and_evaluate function
    train_and_evaluate(
        args,
        X_all, t, y, Z,
        X_train, t_train, y_train, Z_train,
        X_test, t_test, y_test, Z_test,
        true_ite_train, true_ite_test,
        XZ_train, t_train_np, y_train_np,
        dataset=dataset_flag,
        **extra_args
    )

    # Print the true causal effects on the test set
    print("\nTrue causal effects on the test set:")
    if true_ite_test is not None:
        print("True ATE (test):", np.mean(true_ite_test))
        print("True ITE (test) snippet:", true_ite_test[:10])
    else:
        print("No true ITE available for this dataset.")

if __name__ == "__main__":
    main()
