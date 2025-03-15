#!/usr/bin/env python3
"""
train_test.py

This module implements training and evaluation pipelines for causal effect estimation
using both CEVAE (with latent confounder Z) and several baseline models.
In this updated version, the model is trained on the full training set.
For evaluation, we perform K-fold cross-validation on the test set to estimate performance.
"""

import numpy as np
import torch
import pyro
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

from utils import evaluate_ite
from models import CEVAEWithZ

# --- Cross-validation functions on test set ---

def cross_validate_cevae_twins_test(
    X_test: torch.Tensor,
    t_test: torch.Tensor,
    y_test: torch.Tensor,
    Z_test: torch.Tensor,
    true_ite_test: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_decay: float,
    weight_decay: float,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    cv_folds: int = 5,
    seed: int = 0,
) -> dict:
    """
    Performs K-fold cross-validation for the CEVAE model on the test set of the Twins dataset.
    Splitting is done at the twin-pair level (using both twins per pair for evaluation).
    For each fold, the predicted ITE is computed as the difference between the estimated outcomes
    for twin1 and twin0.
    """
    total_samples = X_test.shape[0]
    N_pairs = total_samples // 2
    pair_indices = np.arange(N_pairs)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []

    for _, val_pairs in kf.split(pair_indices):
        # Get indices for each twin in the pair.
        val_twin0 = 2 * val_pairs
        val_twin1 = 2 * val_pairs + 1

        # Compute the true ITE for each pair.
        true_ite_val = (y_test[val_twin1] - y_test[val_twin0]).numpy()

        new_feature_dim = X_test.shape[1] + Z_test.shape[1]
        cevae = CEVAEWithZ(
            feature_dim=new_feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            outcome_dist="normal",
            num_samples=10,
        )
        # Compute predicted outcomes for twin0 and twin1 separately.
        est_ite_twin0 = cevae.ite(X_test[val_twin0], Z_test[val_twin0]).detach().cpu().numpy()
        est_ite_twin1 = cevae.ite(X_test[val_twin1], Z_test[val_twin1]).detach().cpu().numpy()
        # Predicted ITE is the difference between twin1 and twin0.
        est_ite_val = est_ite_twin1 - est_ite_twin0

        ate, pehe, ate_abs_error = evaluate_ite(true_ite_val, est_ite_val)
        results.append((ate, pehe, ate_abs_error))

    avg_results = {
        "ATE_mean": np.mean([r[0] for r in results]),
        "ATE_std": np.std([r[0] for r in results]),
        "PEHE_mean": np.mean([r[1] for r in results]),
        "PEHE_std": np.std([r[1] for r in results]),
        "ATE_Abs_Error_mean": np.mean([r[2] for r in results]),
        "ATE_Abs_Error_std": np.std([r[2] for r in results]),
    }
    return avg_results

def cross_validate_cevae_ihdp_test(
    X_test: torch.Tensor,
    t_test: torch.Tensor,
    y_test: torch.Tensor,
    y_cf_test: torch.Tensor,
    Z_test: torch.Tensor,
    true_ite_test: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_decay: float,
    weight_decay: float,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    cv_folds: int = 5,
    seed: int = 0,
) -> dict:
    """
    Performs K-fold cross-validation for the CEVAE model on the test set of the IHDP dataset.
    For IHDP, true ITE is computed as:
        if t==1: y_test - y_cf_test else: y_cf_test - y_test.
    """
    total_samples = X_test.shape[0]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []

    for _, val_idx in kf.split(np.arange(total_samples)):
        X_val = X_test[val_idx]
        Z_val = Z_test[val_idx]
        t_val = t_test[val_idx]
        y_val = y_test[val_idx]
        y_cf_val = y_cf_test[val_idx]
        true_ite_val = np.where(t_val.numpy() == 1,
                                (y_val - y_cf_val).numpy(),
                                (y_cf_val - y_val).numpy())

        original_feature_dim = X_test.shape[1]
        new_feature_dim = original_feature_dim + Z_test.shape[1]
        cevae = CEVAEWithZ(
            feature_dim=new_feature_dim,
            outcome_dist="normal",
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=10,
        )
        est_ite_val = cevae.ite(X_val, Z_val).detach().cpu().numpy()
        ate, pehe, ate_abs_error = evaluate_ite(true_ite_val, est_ite_val)
        results.append((ate, pehe, ate_abs_error))

    avg_results = {
        "ATE_mean": np.mean([r[0] for r in results]),
        "ATE_std": np.std([r[0] for r in results]),
        "PEHE_mean": np.mean([r[1] for r in results]),
        "PEHE_std": np.std([r[1] for r in results]),
        "ATE_Abs_Error_mean": np.mean([r[2] for r in results]),
        "ATE_Abs_Error_std": np.std([r[2] for r in results]),
    }
    return avg_results

def train_and_evaluate(
    args,
    X: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    Z: torch.Tensor,
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    y_train: torch.Tensor,
    Z_train: torch.Tensor,
    X_test: torch.Tensor,
    t_test: torch.Tensor,
    y_test: torch.Tensor,
    Z_test: torch.Tensor,
    true_ite_train: np.ndarray,
    true_ite_test: np.ndarray,
    XZ_train: np.ndarray,
    t_train_np: np.ndarray,
    y_train_np: np.ndarray,
    dataset: str,
    **kwargs,
) -> None:
    """
    Executes the training and evaluation pipeline.
    The model is trained on the full training set.
    Then, cross-validation is performed on the test set to obtain performance metrics.
    For IHDP, y_cf_test is also used to compute the true ITE.
    """
    if args.cuda:
        torch.set_default_device("cuda")
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    # Train model on full training set.
    new_feature_dim = X.shape[1] + Z.shape[1] 
    if dataset == "ihdp":
        cevae = CEVAEWithZ(
            feature_dim=new_feature_dim,
            outcome_dist="normal",
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_samples=10,
        )
    else:
        cevae = CEVAEWithZ(
            feature_dim=new_feature_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_samples=10,
        )

    cevae.fit(
        X_train, t_train, y_train, z=Z_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        weight_decay=args.weight_decay,
    )

    # Evaluate using cross-validation on the test set.
    print("\nCross-validation on Test Set Evaluation:")
    if dataset == "twins":
        cv_results_cevae = cross_validate_cevae_twins_test(
            X_test, t_test, y_test, Z_test, true_ite_test,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            weight_decay=args.weight_decay,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        print("CEVAE Test CV (Twins):", cv_results_cevae)
    else:  # ihdp
        cv_results_cevae = cross_validate_cevae_ihdp_test(
            X_test, t_test, y_test, kwargs["y_cf_test"], Z_test, true_ite_test,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            weight_decay=args.weight_decay,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        print("CEVAE Test CV (IHDP):", cv_results_cevae)

    print("\nTrue causal effects on the test set:")
    print("True ATE (test):", np.mean(true_ite_test))
    print("True ITE (test) snippet:", true_ite_test[:10])


if __name__ == "__main__":
    # This file is meant to be imported and called by main.py
    pass
