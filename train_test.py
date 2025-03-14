"""
train_test.py

This module implements training, cross-validation, and evaluation pipelines for causal effect estimation
using both CEVAE (with latent confounder Z) and several baseline models.
It contains functions for K-fold cross-validation on both Twins and IHDP datasets.
"""

import numpy as np
import torch
import pyro
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from utils import (
    estimate_ate_ipw,
    estimate_ite_dml,
    estimate_ite_xlearner,
    evaluate_ite,
    estimate_ite_direct,
)
from models import CEVAEWithZ

# --- Cross-validation functions for Twins ---

def cross_validate_models_twins(
    XZ: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    true_ite: np.ndarray,
    K: int = 5,
    seed: int = 0,
) -> dict:
    """
    Performs K-fold cross-validation for non-CEVAE models on the Twins dataset.
    
    Args:
        XZ: Concatenated covariate and Z features.
        t, y: Treatment and outcome arrays.
        true_ite: True individual treatment effects.
        K: Number of folds.
        seed: Random seed.
    
    Returns:
        Dictionary with average performance metrics per model.
    """
    N = XZ.shape[0]
    N_pairs = N // 2
    pair_indices = np.arange(N_pairs)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    results_cv = {
        "XGBoost": [],
        "SVM": [],
        "KNN": [],
        "IPW": [],
        "DML": [],
        "X-Learner": []
    }

    for train_pairs, val_pairs in kf.split(pair_indices):
        # Map pair indices to twin indices
        train_twin0 = 2 * train_pairs
        train_twin1 = 2 * train_pairs + 1
        val_twin0 = 2 * val_pairs
        val_twin1 = 2 * val_pairs + 1

        XZ_train_pairs = XZ[train_twin0]
        y_train_ite = (y[train_twin1] - y[train_twin0]).ravel()

        XZ_val_pairs = XZ[val_twin0]
        y_true_val = (y[val_twin1] - y[val_twin0]).ravel()

        # XGBoost baseline
        est_ite_xgb = estimate_ite_direct(
            XGBRegressor, XZ_train_pairs, y_train_ite, XZ_val_pairs
        )
        ate_xgb, pehe_xgb, ate_abs_error_xgb = evaluate_ite(y_true_val, est_ite_xgb)
        results_cv["XGBoost"].append((ate_xgb, pehe_xgb, ate_abs_error_xgb))

        # SVM baseline
        est_ite_svm = estimate_ite_direct(
            SVR, XZ_train_pairs, y_train_ite, XZ_val_pairs,
            kernel='rbf', C=1.0
        )
        ate_svm, pehe_svm, ate_abs_error_svm = evaluate_ite(y_true_val, est_ite_svm)
        results_cv["SVM"].append((ate_svm, pehe_svm, ate_abs_error_svm))

        # KNN baseline
        est_ite_knn = estimate_ite_direct(
            KNeighborsRegressor, XZ_train_pairs, y_train_ite, XZ_val_pairs,
            n_neighbors=5
        )
        ate_knn, pehe_knn, ate_abs_error_knn = evaluate_ite(y_true_val, est_ite_knn)
        results_cv["KNN"].append((ate_knn, pehe_knn, ate_abs_error_knn))

        # IPW estimation
        XZ_train_full = np.concatenate([XZ[train_twin0], XZ[train_twin1]], axis=0)
        t_train_full = np.concatenate([t[train_twin0], t[train_twin1]])
        y_train_full = np.concatenate([y[train_twin0], y[train_twin1]])
        ate_ipw = estimate_ate_ipw(XZ_train_full, t_train_full, y_train_full)
        ate_true_ipw = np.mean(y_true_val)
        ate_abs_error_ipw = np.abs(ate_ipw - ate_true_ipw)
        results_cv["IPW"].append((ate_ipw, None, ate_abs_error_ipw))

        # DML estimation
        est_ite_dml_val = estimate_ite_dml(XZ_train_full, t_train_full, y_train_full, XZ_val_pairs)
        ate_dml, pehe_dml, ate_abs_error_dml = evaluate_ite(y_true_val, est_ite_dml_val)
        results_cv["DML"].append((ate_dml, pehe_dml, ate_abs_error_dml))

        # X-Learner estimation
        est_ite_xl = estimate_ite_xlearner(XZ_train_full, t_train_full, y_train_full, XZ_val_pairs)
        ate_xl, pehe_xl, ate_abs_error_xl = evaluate_ite(y_true_val, est_ite_xl)
        results_cv["X-Learner"].append((ate_xl, pehe_xl, ate_abs_error_xl))

    avg_results_cv = {}
    for model, vals in results_cv.items():
        ates = [v[0] for v in vals]
        pehes = [v[1] for v in vals if v[1] is not None]
        ate_abs_errors = [v[2] for v in vals]
        avg_results_cv[model] = {
            "ATE_mean": np.mean(ates),
            "ATE_std": np.std(ates),
            "ATE_Abs_Error_mean": np.mean(ate_abs_errors),
            "ATE_Abs_Error_std": np.std(ate_abs_errors),
            "PEHE_mean": np.mean(pehes) if pehes else None,
            "PEHE_std": np.std(pehes) if pehes else None,
        }
    return avg_results_cv


def cross_validate_cevae_twins(
    X: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    Z: torch.Tensor,
    true_ite: np.ndarray,
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
    Performs K-fold cross-validation for the CEVAE model on the Twins dataset.
    Splitting is done at the pair level to respect the paired structure.
    
    Args:
        X, t, y, Z: Full dataset tensors.
        true_ite: True ITE values computed from twin differences.
        Other parameters: training hyperparameters.
    
    Returns:
        Dictionary with averaged ATE, PEHE, and absolute ATE error.
    """
    total_samples = X.shape[0]
    N_pairs = total_samples // 2
    pair_indices = np.arange(N_pairs)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []

    for train_pairs, val_pairs in kf.split(pair_indices):
        train_twin0 = 2 * train_pairs
        train_twin1 = 2 * train_pairs + 1
        val_twin0 = 2 * val_pairs
        val_twin1 = 2 * val_pairs + 1

        X_train = torch.cat([X[train_twin0], X[train_twin1]], dim=0)
        t_train = torch.cat([t[train_twin0], t[train_twin1]], dim=0)
        y_train = torch.cat([y[train_twin0], y[train_twin1]], dim=0)
        Z_train = torch.cat([Z[train_twin0], Z[train_twin1]], dim=0)

        X_val = X[val_twin0]  # use one twin from each pair for validation
        Z_val = Z[val_twin0]
        # True ITE for validation: difference of paired outcomes
        true_ite_val = (y[val_twin1] - y[val_twin0]).numpy()

        new_feature_dim = X.shape[1] + Z.shape[1]

        cevae = CEVAEWithZ(
            feature_dim=new_feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            outcome_dist="normal",
            num_samples=10,
        )

        cevae.fit(
            X_train, t_train, y_train, z=Z_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            weight_decay=weight_decay,
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


# --- Cross-validation functions for IHDP ---

def estimate_ite_direct_ihdp(model_class, X_train, y_train, t_train, X_val, **model_kwargs):
    """
    Direct ITE estimation for IHDP using two separate models (one for treated, one for control).
    """
    treated_idx = (t_train == 1).squeeze().numpy().astype(bool)
    control_idx = (t_train == 0).squeeze().numpy().astype(bool)
    model_t = model_class(**model_kwargs)
    model_c = model_class(**model_kwargs)
    model_t.fit(X_train[treated_idx], y_train[treated_idx])
    model_c.fit(X_train[control_idx], y_train[control_idx])
    pred_t = model_t.predict(X_val)
    pred_c = model_c.predict(X_val)
    return pred_t - pred_c


def cross_validate_models_ihdp(
    XZ: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    y_cf: np.ndarray,
    true_ite: np.ndarray,
    K: int = 5,
    seed: int = 0,
) -> dict:
    """
    K-fold cross-validation for non-CEVAE models on the IHDP dataset.
    
    Args:
        XZ: Combined covariate and latent confounder features.
        t, y, y_cf: Treatment, factual outcome, and counterfactual outcome.
        true_ite: True ITE computed as (y - y_cf) for t==1 and (y_cf - y) for t==0.
    
    Returns:
        Dictionary of average performance metrics for each model.
    """
    total_samples = XZ.shape[0]
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    results_cv = {
        "Direct": [],
        "XGBoost": [],
        "SVM": [],
        "KNN": [],
        "IPW": [],
        "DML": [],
        "X-Learner": []
    }

    for train_idx, val_idx in kf.split(np.arange(total_samples)):
        X_train = XZ[train_idx]
        t_train = t[train_idx]
        y_train = y[train_idx]
        X_val = XZ[val_idx]
        true_ite_val = true_ite[val_idx]

        est_ite_direct = estimate_ite_direct_ihdp(
            XGBRegressor, X_train, y_train, t_train, X_val
        )
        ate_direct, pehe_direct, ate_abs_error_direct = evaluate_ite(true_ite_val, est_ite_direct)
        results_cv["Direct"].append((ate_direct, pehe_direct, ate_abs_error_direct))

        est_ite_xgb = estimate_ite_direct_ihdp(
            XGBRegressor, X_train, y_train, t_train, X_val
        )
        ate_xgb, pehe_xgb, ate_abs_error_xgb = evaluate_ite(true_ite_val, est_ite_xgb)
        results_cv["XGBoost"].append((ate_xgb, pehe_xgb, ate_abs_error_xgb))

        est_ite_svm = estimate_ite_direct_ihdp(
            SVR, X_train, y_train, t_train, X_val, kernel='rbf', C=1.0
        )
        ate_svm, pehe_svm, ate_abs_error_svm = evaluate_ite(true_ite_val, est_ite_svm)
        results_cv["SVM"].append((ate_svm, pehe_svm, ate_abs_error_svm))

        est_ite_knn = estimate_ite_direct_ihdp(
            KNeighborsRegressor, X_train, y_train, t_train, X_val, n_neighbors=5
        )
        ate_knn, pehe_knn, ate_abs_error_knn = evaluate_ite(true_ite_val, est_ite_knn)
        results_cv["KNN"].append((ate_knn, pehe_knn, ate_abs_error_knn))

        ate_ipw = estimate_ate_ipw(X_train, t_train, y_train)
        ate_true_ipw = np.mean(true_ite_val)
        ate_abs_error_ipw = np.abs(ate_ipw - ate_true_ipw)
        results_cv["IPW"].append((ate_ipw, None, ate_abs_error_ipw))

        est_ite_dml = estimate_ite_dml(X_train, t_train, y_train, X_val)
        ate_dml, pehe_dml, ate_abs_error_dml = evaluate_ite(true_ite_val, est_ite_dml)
        results_cv["DML"].append((ate_dml, pehe_dml, ate_abs_error_dml))

        est_ite_xl = estimate_ite_xlearner(X_train, t_train, y_train, X_val)
        ate_xl, pehe_xl, ate_abs_error_xl = evaluate_ite(true_ite_val, est_ite_xl)
        results_cv["X-Learner"].append((ate_xl, pehe_xl, ate_abs_error_xl))

    avg_results_cv = {}
    for model, vals in results_cv.items():
        ates = [v[0] for v in vals]
        pehes = [v[1] for v in vals if v[1] is not None]
        ate_abs_errors = [v[2] for v in vals]
        avg_results_cv[model] = {
            "ATE_mean": np.mean(ates),
            "ATE_std": np.std(ates),
            "ATE_Abs_Error_mean": np.mean(ate_abs_errors),
            "ATE_Abs_Error_std": np.std(ate_abs_errors),
            "PEHE_mean": np.mean(pehes) if pehes else None,
            "PEHE_std": np.std(pehes) if pehes else None,
        }
    return avg_results_cv


def cross_validate_cevae_ihdp(
    X: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    y_cf: torch.Tensor,
    Z: torch.Tensor,
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
    Performs K-fold cross-validation for the CEVAE model on the IHDP dataset.
    The true ITE is computed as:
        if t == 1:  y - y_cf
        else:       y_cf - y
    """
    total_samples = X.shape[0]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []

    for train_idx, val_idx in kf.split(np.arange(total_samples)):
        X_train = X[train_idx]
        t_train = t[train_idx]
        y_train = y[train_idx]
        Z_train = Z[train_idx]

        X_val = X[val_idx]
        Z_val = Z[val_idx]
        y_val = y[val_idx]
        y_cf_val = y_cf[val_idx]
        t_val = t[val_idx]
        true_ite_val = np.where(t_val.numpy() == 1,
                                (y_val - y_cf_val).numpy(),
                                (y_cf_val - y_val).numpy())

        original_feature_dim = X.shape[1]
        z_dim = Z.shape[1]
        new_feature_dim = original_feature_dim + z_dim

        cevae = CEVAEWithZ(
            feature_dim=new_feature_dim,
            outcome_dist="normal",
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=10,
        )

        cevae.fit(
            X_train, t_train, y_train, z=Z_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            weight_decay=weight_decay,
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
    Executes the training, cross-validation, and final evaluation pipeline.
    Branches based on the dataset type ('twins' or 'ihdp').
    """
    if args.cuda:
        torch.set_default_device("cuda")
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    if dataset == "twins":
        cv_results_cevae = cross_validate_cevae_twins(
            X, t, y, Z, true_ite_train,
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
        print("\nCross-validation results for CEVAE on training set (Twins):")
        print(f"CEVAE: ATE={cv_results_cevae['ATE_mean']:.3f}±{cv_results_cevae['ATE_std']:.3f}", end='')
        print(f", ATE Abs Error={cv_results_cevae['ATE_Abs_Error_mean']:.3f}±{cv_results_cevae['ATE_Abs_Error_std']:.3f}", end='')
        print(f", PEHE={cv_results_cevae['PEHE_mean']:.3f}±{cv_results_cevae['PEHE_std']:.3f}")

        if not args.only_cevae:
            cv_results = cross_validate_models_twins(
                XZ_train, t_train_np, y_train_np, true_ite_train,
                K=args.cv_folds, seed=args.seed
            )
            print("\nCross-validation results for non-CEVAE models on training set (Twins):")
            for model, res in cv_results.items():
                print(f"{model}: ATE={res['ATE_mean']:.3f}±{res['ATE_std']:.3f}", end='')
                print(f", ATE Abs Error={res['ATE_Abs_Error_mean']:.3f}±{res['ATE_Abs_Error_std']:.3f}", end='')
                if res['PEHE_mean'] is not None:
                    print(f", PEHE={res['PEHE_mean']:.3f}±{res['PEHE_std']:.3f}")
                else:
                    print(", PEHE=Not applicable")
    else:  # ihdp
        cv_results_cevae = cross_validate_cevae_ihdp(
            X, t, y, y, Z,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            weight_decay=args.weight_decay,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cv_folds=args.cv_folds,
            seed=args.seed
        )
        print("\nCross-validation results for CEVAE on training set (IHDP):")
        print(f"CEVAE: ATE={cv_results_cevae['ATE_mean']:.3f}±{cv_results_cevae['ATE_std']:.3f}", end='')
        print(f", ATE Abs Error={cv_results_cevae['ATE_Abs_Error_mean']:.3f}±{cv_results_cevae['ATE_Abs_Error_std']:.3f}", end='')
        print(f", PEHE={cv_results_cevae['PEHE_mean']:.3f}±{cv_results_cevae['PEHE_std']:.3f}")

        if not args.only_cevae:
            cv_results = cross_validate_models_ihdp(
                XZ_train, t_train_np, y_train_np, y_train_np, true_ite_train,
                K=args.cv_folds, seed=args.seed
            )
            print("\nCross-validation results for non-CEVAE models on training set (IHDP):")
            for model, res in cv_results.items():
                print(f"{model}: ATE={res['ATE_mean']:.3f}±{res['ATE_std']:.3f}", end='')
                print(f", ATE Abs Error={res['ATE_Abs_Error_mean']:.3f}±{res['ATE_Abs_Error_std']:.3f}", end='')
                if res['PEHE_mean'] is not None:
                    print(f", PEHE={res['PEHE_mean']:.3f}±{res['PEHE_std']:.3f}")
                else:
                    print(", PEHE=Not applicable")

    # Final training on full training set and evaluation on test set.
    new_feature_dim = X.shape[1] + Z.shape[1] if dataset == "ihdp" else X.shape[1]
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

    print("\nTest Set Evaluation:")
    if dataset == "twins":
        # For Twins, construct XZ from unshifted X and Z
        XZ_full = np.concatenate([X.numpy(), Z.numpy()], axis=1)
        XZ0_train_full = XZ_full[kwargs["train_twin0"]]
        ITE_train_full = (y[kwargs["train_twin1"]] - y[kwargs["train_twin0"]]).numpy()
        XZ0_test_full = XZ_full[kwargs["test_twin0"]]

        est_ite_cevae_test = cevae.ite(X[kwargs["test_twin0"]], Z[kwargs["test_twin0"]]).detach().cpu().numpy()
        ate_cevae_test, pehe_cevae_test, ate_abs_error_cevae_test = evaluate_ite(true_ite_test, est_ite_cevae_test)
        print(f"CEVAE: ATE={ate_cevae_test:.3f}, ATE Abs Error={ate_abs_error_cevae_test:.3f}, PEHE={pehe_cevae_test:.3f}")

        if not args.only_cevae:
            xgb_model = XGBRegressor()
            xgb_model.fit(XZ0_train_full, ITE_train_full)
            est_ite_xgb_test = xgb_model.predict(XZ0_test_full)
            ate_xgb_test, pehe_xgb_test, ate_abs_error_xgb_test = evaluate_ite(true_ite_test, est_ite_xgb_test)
            print("\nTest Set Results (Other Models):")
            print(f"XGBoost: ATE={ate_xgb_test:.3f}, ATE Abs Error={ate_abs_error_xgb_test:.3f}, PEHE={pehe_xgb_test:.3f}")
            # Similar evaluations can be added for SVM, KNN, etc.
    else:  # ihdp
        est_ite_cevae_test = cevae.ite(X_test, Z_test).detach().cpu().numpy()
        # after `est_ite_cevae_test = cevae.ite(X_test, Z_test)...`
        if true_ite_test is not None:
            ate_cevae_test, pehe_cevae_test, ate_abs_error_cevae_test = evaluate_ite(true_ite_test, est_ite_cevae_test)
            print(f"CEVAE: ATE={ate_cevae_test:.3f}, ATE Abs Error={ate_abs_error_cevae_test:.3f}, PEHE={pehe_cevae_test:.3f}")
        else:
            print("CEVAE predictions on test set computed, but no true ITE available for IHDP to compare.")

        if not args.only_cevae:
            # Direct estimation baseline using XGBoost for IHDP
            direct_model = XGBRegressor()
            direct_model.fit(XZ_train, y_train_np)
            est_ite_direct_test = direct_model.predict(X_test.numpy())
            ate_direct_test, pehe_direct_test, ate_abs_error_direct_test = evaluate_ite(true_ite_test, est_ite_direct_test)
            print("\nTest Set Results (Direct Estimation with XGB):")
            print(f"Direct: ATE={ate_direct_test:.3f}, ATE Abs Error={ate_abs_error_direct_test:.3f}, PEHE={pehe_direct_test:.3f}")

    # Optionally print true causal effects.
    print("\nTrue causal effects on the test set:")
    print("True ATE (test):", np.mean(true_ite_test))
    print("True ITE (test) snippet:", true_ite_test[:10])


if __name__ == "__main__":
    # This file is meant to be imported and called by main.py
    pass
