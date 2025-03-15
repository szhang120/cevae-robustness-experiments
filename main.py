#!/usr/bin/env python3
import argparse
import logging
import torch
import pandas as pd
import numpy as np
import os
import io
from contextlib import redirect_stdout

# data-loading
from data import load_ihdp_data, prepare_train_test_split_ihdp
from data import load_twins_data, prepare_train_test_split

# training/evaluation
from train_test import train_and_evaluate

def set_default_hyperparams(args):
    if args.dataset == "twins":
        if args.num_data is None: args.num_data = 23968
        if args.feature_dim is None: args.feature_dim = 44
        if args.num_epochs is None: args.num_epochs = 30
        if args.latent_dim is None: args.latent_dim = 30
        if args.hidden_dim is None: args.hidden_dim = 300
        if args.num_layers is None: args.num_layers = 2
        if args.batch_size is None: args.batch_size = 512
        if args.learning_rate is None: args.learning_rate = 1e-3
        if args.learning_rate_decay is None: args.learning_rate_decay = 0.95
        if args.weight_decay is None: args.weight_decay = 1e-4
    else:  # ihdp
        if args.feature_dim is None: args.feature_dim = 24
        if args.num_epochs is None: args.num_epochs = 50
        if args.latent_dim is None: args.latent_dim = 30
        if args.hidden_dim is None: args.hidden_dim = 150
        if args.num_layers is None: args.num_layers = 2
        if args.batch_size is None: args.batch_size = 256
        if args.learning_rate is None: args.learning_rate = 1e-3
        if args.learning_rate_decay is None: args.learning_rate_decay = 0.995
        if args.weight_decay is None: args.weight_decay = 1e-6
    return args

def comprehensive_evaluation(args):
    # Evaluate both datasets, both shifted and nonshifted, and sweep over flipping probabilities.
    datasets = ["ihdp"]
    shift_options = [False, True]  # non-shifted, shifted
    p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for dataset in datasets:
        log_filename = f"evaluation2_{dataset}.txt"
        with open(log_filename, "w") as log_file:
            log_file.write(f"=== Comprehensive Evaluation for dataset: {dataset} ===\n")
            for train_shifted in shift_options:
                log_file.write(f"\n--- Train shifted = {train_shifted} ---\n")
                for p in p_values:
                    log_file.write(f"\n*** Experiment: flipping probability p = {p} ***\n")
                    if dataset == "ihdp":
                        # Load IHDP data with chosen shifted flag.
                        X_input, t, y, y_cf, Z = load_ihdp_data(train_shifted=train_shifted)
                        # Override latent confounder Z with file corresponding to current p.
                        z_file = f"data/IHDP/processed_Z_ihdp_p{p}.csv"
                        try:
                            Z_df = pd.read_csv(z_file)
                            Z = torch.tensor(Z_df.values, dtype=torch.float)
                            log_file.write(f"Loaded IHDP Z from {z_file}\n")
                        except Exception as e:
                            log_file.write(f"Error loading IHDP Z from {z_file}: {e}\n")
                        (X_train, t_train, y_train, y_cf_train, Z_train,
                         X_test, t_test, y_test, y_cf_test, Z_test,
                         true_ite_train, true_ite_test,
                         XZ_train, t_train_np, y_train_np,
                         train_idx, test_idx) = prepare_train_test_split_ihdp(
                            X_input, t, y, y_cf, Z,
                            test_size=args.test_size,
                            seed=args.seed
                        )
                        extra_args = {"y_cf_test": y_cf_test}
                        data_for_training = X_input
                    else:  # twins
                        X, t, y, Z = load_twins_data(train_shifted=train_shifted)
                        z_file = f"data/TWINS/processed_z_p{p}.csv"
                        try:
                            Z_df = pd.read_csv(z_file)
                            Z = torch.tensor(Z_df.values, dtype=torch.float)
                            log_file.write(f"Loaded Twins Z from {z_file}\n")
                        except Exception as e:
                            log_file.write(f"Error loading Twins Z from {z_file}: {e}\n")
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
                        extra_args = {
                            "train_twin0": train_twin0,
                            "train_twin1": train_twin1,
                            "test_twin0": test_twin0,
                            "test_twin1": test_twin1
                        }
                        data_for_training = X
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        if not args.skip_cevae:
                            train_and_evaluate(
                                args,
                                data_for_training, t, y, Z,
                                X_train, t_train, y_train, Z_train,
                                X_test, t_test, y_test, Z_test,
                                true_ite_train, true_ite_test,
                                XZ_train, t_train_np, y_train_np,
                                dataset=dataset,
                                **extra_args
                            )
                        else:
                            print("Skipping CEVAE training and evaluation as requested (--skip-cevae).")
                        if args.skip_cevae or not args.only_cevae:
                            print("\nRunning baseline evaluations using utility models via 3-fold CV...")
                            from utils import (
                                estimate_ate_ipw_cv,
                                estimate_ite_dml_cv,
                                estimate_ite_xlearner_cv,
                                estimate_ite_svm_cv,
                                estimate_ite_knn_cv,
                                estimate_ite_interaction_cv,
                                estimate_ite_xgb_cv,
                                evaluate_ite,
                            )
                            if dataset == "twins":
                                ate_ipw_cv = estimate_ate_ipw_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                print(f"IPW CV (ATE) for Twins: {ate_ipw_cv:.4f}")
                                dml_cv = estimate_ite_dml_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                dml_cv = np.array(dml_cv).reshape(-1)
                                ate_dml, pehe_dml, ate_abs_dml = evaluate_ite(true_ite_train, dml_cv)
                                print(f"DML CV (Twins): ATE={ate_dml:.4f}, PEHE={pehe_dml:.4f}, ATE_Abs_Error={ate_abs_dml:.4f}")
                                xl_cv = estimate_ite_xlearner_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                xl_cv = np.array(xl_cv).reshape(-1)
                                ate_xl, pehe_xl, ate_abs_xl = evaluate_ite(true_ite_train, xl_cv)
                                print(f"X-Learner CV (Twins): ATE={ate_xl:.4f}, PEHE={pehe_xl:.4f}, ATE_Abs_Error={ate_abs_xl:.4f}")
                                svm_cv = estimate_ite_svm_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                svm_cv = np.array(svm_cv).reshape(-1)
                                ate_svm, pehe_svm, abs_svm = evaluate_ite(true_ite_train, svm_cv)
                                print(f"SVM CV (Twins): ATE={ate_svm:.4f}, PEHE={pehe_svm:.4f}, ATE_Abs_Error={abs_svm:.4f}")
                                knn_cv = estimate_ite_knn_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                knn_cv = np.array(knn_cv).reshape(-1)
                                ate_knn, pehe_knn, abs_knn = evaluate_ite(true_ite_train, knn_cv)
                                print(f"KNN CV (Twins): ATE={ate_knn:.4f}, PEHE={pehe_knn:.4f}, ATE_Abs_Error={abs_knn:.4f}")
                                ilr_cv = estimate_ite_interaction_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                ilr_cv = np.array(ilr_cv).reshape(-1)
                                ate_ilr, pehe_ilr, abs_ilr = evaluate_ite(true_ite_train, ilr_cv)
                                print(f"Interacted LR CV (Twins): ATE={ate_ilr:.4f}, PEHE={pehe_ilr:.4f}, ATE_Abs_Error={abs_ilr:.4f}")
                                xgb_cv = estimate_ite_xgb_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                xgb_cv = np.array(xgb_cv).reshape(-1)
                                ate_xgb, pehe_xgb, abs_xgb = evaluate_ite(true_ite_train, xgb_cv)
                                print(f"XGBoost CV (Twins): ATE={ate_xgb:.4f}, PEHE={pehe_xgb:.4f}, ATE_Abs_Error={abs_xgb:.4f}")
                            else:
                                ate_ipw_cv = estimate_ate_ipw_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                print(f"IPW CV (ATE) for IHDP: {ate_ipw_cv:.4f}")
                                dml_cv = estimate_ite_dml_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                dml_cv = np.array(dml_cv).reshape(-1)
                                ate_dml, pehe_dml, ate_abs_dml = evaluate_ite(true_ite_train, dml_cv)
                                print(f"DML CV (IHDP): ATE={ate_dml:.4f}, PEHE={pehe_dml:.4f}, ATE_Abs_Error={ate_abs_dml:.4f}")
                                xl_cv = estimate_ite_xlearner_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                xl_cv = np.array(xl_cv).reshape(-1)
                                ate_xl, pehe_xl, abs_xl = evaluate_ite(true_ite_train, xl_cv)
                                print(f"X-Learner CV (IHDP): ATE={ate_xl:.4f}, PEHE={pehe_xl:.4f}, ATE_Abs_Error={abs_xl:.4f}")
                                svm_cv = estimate_ite_svm_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                svm_cv = np.array(svm_cv).reshape(-1)
                                ate_svm, pehe_svm, abs_svm = evaluate_ite(true_ite_train, svm_cv)
                                print(f"SVM CV (IHDP): ATE={ate_svm:.4f}, PEHE={pehe_svm:.4f}, ATE_Abs_Error={abs_svm:.4f}")
                                knn_cv = estimate_ite_knn_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                knn_cv = np.array(knn_cv).reshape(-1)
                                ate_knn, pehe_knn, abs_knn = evaluate_ite(true_ite_train, knn_cv)
                                print(f"KNN CV (IHDP): ATE={ate_knn:.4f}, PEHE={pehe_knn:.4f}, ATE_Abs_Error={abs_knn:.4f}")
                                ilr_cv = estimate_ite_interaction_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                ilr_cv = np.array(ilr_cv).reshape(-1)
                                ate_ilr, pehe_ilr, abs_ilr = evaluate_ite(true_ite_train, ilr_cv)
                                print(f"Interacted LR CV (IHDP): ATE={ate_ilr:.4f}, PEHE={pehe_ilr:.4f}, ATE_Abs_Error={abs_ilr:.4f}")
                                xgb_cv = estimate_ite_xgb_cv(XZ_train, t_train_np, y_train_np, cv=args.cv_folds)
                                xgb_cv = np.array(xgb_cv).reshape(-1)
                                ate_xgb, pehe_xgb, abs_xgb = evaluate_ite(true_ite_train, xgb_cv)
                                print(f"XGBoost CV (IHDP): ATE={ate_xgb:.4f}, PEHE={pehe_xgb:.4f}, ATE_Abs_Error={abs_xgb:.4f}")
                    experiment_output = buffer.getvalue()
                    log_file.write(experiment_output)
    print("Comprehensive evaluation completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of datasets (Twins and IHDP) with shifted and non-shifted covariates, across various flipping probabilities."
    )
    parser.add_argument("--dataset", choices=["twins", "ihdp"], default="twins",
                        help="Dataset to use in default mode (overridden in sweep).")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--cv-folds", default=3, type=int)
    parser.add_argument("--num-data", type=int, default=23968)
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-rate-decay", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--only-cevae", action="store_true", default=False)
    parser.add_argument("--skip-cevae", action="store_true", default=False)
    args = parser.parse_args()
    args = set_default_hyperparams(args)

    logging.getLogger("pyro").setLevel(logging.DEBUG)
    if logging.getLogger("pyro").handlers:
        logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

    comprehensive_evaluation(args)

if __name__ == "__main__":
    main()
