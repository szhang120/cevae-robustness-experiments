import numpy as np
import torch
import pyro

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from utils import (
    estimate_ate_ipw,
    estimate_ite_dml,
    estimate_ite_xlearner,
    evaluate_ite,
    estimate_ite_direct
)
from models import CEVAEWithZ


def cross_validate_models(XZ, t, y, y_true, K=5, seed=0):
    """
    Perform K-fold CV
    """
    from sklearn.model_selection import KFold

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
        train_twin0 = 2 * train_pairs
        train_twin1 = 2 * train_pairs + 1
        val_twin0 = 2 * val_pairs
        val_twin1 = 2 * val_pairs + 1

        # for direct-ITE regressions
        XZ_train_pairs = XZ[train_twin0]
        y_train_ite = (y[train_twin1] - y[train_twin0]).ravel()

        XZ_val_pairs = XZ[val_twin0]
        y_true_val = (y[val_twin1] - y[val_twin0]).ravel()

        est_ite_xgb = estimate_ite_direct(
            XGBRegressor, XZ_train_pairs, y_train_ite, XZ_val_pairs
        )
        ate_xgb, pehe_xgb, ate_abs_error_xgb = evaluate_ite(y_true_val, est_ite_xgb)
        results_cv["XGBoost"].append((ate_xgb, pehe_xgb, ate_abs_error_xgb))

        est_ite_svm = estimate_ite_direct(
            SVR, XZ_train_pairs, y_train_ite, XZ_val_pairs,
            kernel='rbf', C=1.0
        )
        ate_svm, pehe_svm, ate_abs_error_svm = evaluate_ite(y_true_val, est_ite_svm)
        results_cv["SVM"].append((ate_svm, pehe_svm, ate_abs_error_svm))

        est_ite_knn = estimate_ite_direct(
            KNeighborsRegressor, XZ_train_pairs, y_train_ite, XZ_val_pairs,
            n_neighbors=5
        )
        ate_knn, pehe_knn, ate_abs_error_knn = evaluate_ite(y_true_val, est_ite_knn)
        results_cv["KNN"].append((ate_knn, pehe_knn, ate_abs_error_knn))

        XZ_train_full = np.concatenate([XZ[train_twin0], XZ[train_twin1]], axis=0)
        t_train_full = np.concatenate([t[train_twin0], t[train_twin1]])
        y_train_full = np.concatenate([y[train_twin0], y[train_twin1]])

        ate_ipw = estimate_ate_ipw(XZ_train_full, t_train_full, y_train_full)
        ate_true_ipw = np.mean(y_true_val)
        ate_abs_error_ipw = np.abs(ate_ipw - ate_true_ipw)
        results_cv["IPW"].append((ate_ipw, None, ate_abs_error_ipw))

        est_ite_dml_val = estimate_ite_dml(XZ_train_full, t_train_full, y_train_full, XZ_val_pairs)
        ate_dml, pehe_dml, ate_abs_error_dml = evaluate_ite(y_true_val, est_ite_dml_val)
        results_cv["DML"].append((ate_dml, pehe_dml, ate_abs_error_dml))

        est_ite_xl = estimate_ite_xlearner(XZ_train_full, t_train_full, y_train_full, XZ_val_pairs)
        ate_xl, pehe_xl, ate_abs_error_xl = evaluate_ite(y_true_val, est_ite_xl)
        results_cv["X-Learner"].append((ate_xl, pehe_xl, ate_abs_error_xl))

    # collect results
    avg_results_cv = {}
    for model, vals in results_cv.items():
        ates = [v[0] for v in vals]
        pehes = [v[1] for v in vals if v[1] is not None]
        ate_abs_errors = [v[2] for v in vals]

        ate_mean = np.mean(ates)
        ate_std = np.std(ates)
        ate_abs_error_mean = np.mean(ate_abs_errors)
        ate_abs_error_std = np.std(ate_abs_errors)

        if len(pehes) > 0:
            pehe_mean = np.mean(pehes)
            pehe_std = np.std(pehes)
        else:
            pehe_mean = None
            pehe_std = None

        avg_results_cv[model] = {
            "ATE_mean": ate_mean,
            "ATE_std": ate_std,
            "ATE_Abs_Error_mean": ate_abs_error_mean,
            "ATE_Abs_Error_std": ate_abs_error_std,
            "PEHE_mean": pehe_mean,
            "PEHE_std": pehe_std
        }

    return avg_results_cv


def train_and_evaluate(args, 
                       X, t, y, Z, 
                       X_train, t_train, y_train, Z_train,
                       X_test, t_test, y_test, Z_test,
                       true_ite_train, true_ite_test,
                       XZ_train, t_train_np, y_train_np,
                       train_twin0, train_twin1,
                       test_twin0, test_twin1):
    """
    Execute the entire training, validation (via CV), and testing pipeline.
    """
    if args.cuda:
        torch.set_default_device("cuda")
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    # cross validation
    cv_results = cross_validate_models(
        XZ_train, t_train_np, y_train_np, true_ite_train,
        K=args.cv_folds, seed=args.seed
    )
    print("\nCross-validation results on training set:")
    for model, res in cv_results.items():
        print(f"{model}: ATE={res['ATE_mean']:.3f}±{res['ATE_std']:.3f}", end='')
        print(f", ATE Abs Error={res['ATE_Abs_Error_mean']:.3f}±{res['ATE_Abs_Error_std']:.3f}", end='')
        if res['PEHE_mean'] is not None:
            print(f", PEHE={res['PEHE_mean']:.3f}±{res['PEHE_std']:.3f}")
        else:
            print(", PEHE=Not applicable")

    # train the CEVAE
    original_feature_dim = X.shape[1]
    z_dim = Z.shape[1]
    new_feature_dim = original_feature_dim + z_dim

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

    print("\n Test Set Evaluation:")

    XZ = np.concatenate([X.numpy(), Z.numpy()], axis=1)
    XZ0_train_full = XZ[train_twin0]
    ITE_train_full = (y[train_twin1] - y[train_twin0]).numpy()
    XZ0_test_full = XZ[test_twin0]

    est_ite_cevae_test = cevae.ite(X[test_twin0], Z[test_twin0]).detach().cpu().numpy()
    ate_cevae_test, pehe_cevae_test, ate_abs_error_cevae_test = evaluate_ite(true_ite_test, est_ite_cevae_test)

    xgb_model = XGBRegressor()
    xgb_model.fit(XZ0_train_full, ITE_train_full)
    est_ite_xgb_test = xgb_model.predict(XZ0_test_full)
    ate_xgb_test, pehe_xgb_test, ate_abs_error_xgb_test = evaluate_ite(true_ite_test, est_ite_xgb_test)

    svm_model = SVR(kernel='rbf', C=1.0)
    svm_model.fit(XZ0_train_full, ITE_train_full)
    est_ite_svm_test = svm_model.predict(XZ0_test_full)
    ate_svm_test, pehe_svm_test, ate_abs_error_svm_test = evaluate_ite(true_ite_test, est_ite_svm_test)

    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(XZ0_train_full, ITE_train_full)
    est_ite_knn_test = knn_model.predict(XZ0_test_full)
    ate_knn_test, pehe_knn_test, ate_abs_error_knn_test = evaluate_ite(true_ite_test, est_ite_knn_test)

    ate_ipw_test = estimate_ate_ipw(XZ_train, t_train_np, y_train_np)
    ate_true_ipw_test = np.mean(true_ite_test)
    ate_abs_error_ipw_test = np.abs(ate_ipw_test - ate_true_ipw_test)

    est_ite_dml_test = estimate_ite_dml(XZ_train, t_train_np, y_train_np, XZ0_test_full)
    ate_dml_test, pehe_dml_test, ate_abs_error_dml_test = evaluate_ite(true_ite_test, est_ite_dml_test)

    est_ite_xl_test = estimate_ite_xlearner(XZ_train, t_train_np, y_train_np, XZ0_test_full)
    ate_xl_test, pehe_xl_test, ate_abs_error_xl_test = evaluate_ite(true_ite_test, est_ite_xl_test)

    print("\nTest Set Results:")
    results_test = {
        "CEVAE": (ate_cevae_test, pehe_cevae_test, ate_abs_error_cevae_test),
        "XGBoost": (ate_xgb_test, pehe_xgb_test, ate_abs_error_xgb_test),
        "SVM": (ate_svm_test, pehe_svm_test, ate_abs_error_svm_test),
        "KNN": (ate_knn_test, pehe_knn_test, ate_abs_error_knn_test),
        "IPW": (ate_ipw_test, None, ate_abs_error_ipw_test),
        "DML": (ate_dml_test, pehe_dml_test, ate_abs_error_dml_test),
        "X-Learner": (ate_xl_test, pehe_xl_test, ate_abs_error_xl_test)
    }

    for model, (ate_val, pehe_val, ate_abs_error_val) in results_test.items():
        print(f"{model}: ATE={ate_val:.3f}", end='')
        print(f", ATE Abs Error={ate_abs_error_val:.3f}", end='')
        if pehe_val is not None:
            print(f", PEHE={pehe_val:.3f}")
        else:
            print(", PEHE=N/A")