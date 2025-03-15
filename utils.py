import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from econml.dml import DML
from econml.metalearners import XLearner
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

from sklearn.model_selection import KFold


##############################################################################
# 3-fold CV wrappers for every model (for causal baselines)
##############################################################################

def estimate_ate_ipw_cv(XZ, t, y, cv=3):
    """
    Estimate ATE via IPW using 3-fold cross-validation.
    Since IPW naturally estimates the overall ATE (not individual treatment effects),
    we perform CV: in each fold, the propensity model is trained on the training set,
    then ATE is computed on the validation set. The final ATE is the weighted average.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    ate_folds = []
    fold_sizes = []
    for train_index, val_index in kf.split(XZ):
        XZ_train, XZ_val = XZ[train_index], XZ[val_index]
        t_train, t_val = t[train_index], t[val_index]
        y_val = y[val_index]
        scaler = StandardScaler()
        XZ_train_scaled = scaler.fit_transform(XZ_train)
        XZ_val_scaled = scaler.transform(XZ_val)
        lr = LogisticRegression(max_iter=10000)
        lr.fit(XZ_train_scaled, t_train)
        p_val = lr.predict_proba(XZ_val_scaled)[:, 1]
        p_val = np.clip(p_val, 1e-5, 1 - 1e-5)
        ate_fold = np.mean(y_val[t_val == 1] / p_val[t_val == 1]) - np.mean(y_val[t_val == 0] / (1 - p_val[t_val == 0]))
        ate_folds.append(ate_fold)
        fold_sizes.append(len(val_index))
    total = np.sum(fold_sizes)
    ate_cv = np.sum(np.array(ate_folds) * np.array(fold_sizes)) / total
    return ate_cv


def estimate_ite_dml_cv(XZ, t, y, cv=3):
    """
    Estimate ITE via DML using 3-fold cross-validation.
    For each fold, a DML instance is trained on the training data and then used to
    predict individual treatment effects on the validation fold.
    The ITE predictions are then aggregated (in original order) and returned.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    ite_pred = np.empty(XZ.shape[0])
    for train_index, val_index in kf.split(XZ):
        XZ_train, XZ_val = XZ[train_index], XZ[val_index]
        t_train, t_val = t[train_index], t[val_index]
        y_train = y[train_index]
        dml = DML(
            model_y=RandomForestRegressor(random_state=0),
            model_t=RandomForestClassifier(random_state=0),
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            discrete_treatment=True
        )
        dml.fit(y_train, t_train, X=XZ_train)
        ite_pred[val_index] = dml.effect(XZ_val, T0=0, T1=1)
    return ite_pred


def estimate_ite_xlearner_cv(XZ, t, y, cv=3):
    """
    Estimate ITE via X-Learner using 3-fold cross-validation.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    ite_pred = np.empty(XZ.shape[0])
    for train_index, val_index in kf.split(XZ):
        XZ_train, XZ_val = XZ[train_index], XZ[val_index]
        t_train, t_val = t[train_index], t[val_index]
        y_train = y[train_index]
        base_learner = RandomForestRegressor(random_state=0)
        xlearner = XLearner(models=base_learner)
        xlearner.fit(y_train, t_train, X=XZ_train)
        ite_pred[val_index] = xlearner.effect(XZ_val)
    return ite_pred


##############################################################################
# Traditional (non-causal) baseline CV for SVM, KNN, and XGBoost
##############################################################################

def estimate_ite_traditional_cv(model_class, X, t, y, cv=3, **model_kwargs):
    """
    Estimate ITE using traditional k-fold cross validation.
    In each fold, a regression model is trained on the entire training data,
    where the features are concatenated with the treatment indicator.
    For each validation sample, two predictions are made:
      - One with treatment set to 1.
      - One with treatment set to 0.
    The difference of these predictions is taken as the estimated ITE.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    ite_pred = np.empty(X.shape[0])
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        t_train, t_val = t[train_index], t[val_index]
        y_train = y[train_index]
        # Train on training data: concatenate treatment indicator.
        X_train_full = np.concatenate([X_train, t_train.reshape(-1, 1)], axis=1)
        model = model_class(**model_kwargs)
        model.fit(X_train_full, y_train)
        # For validation, prepare two versions: one with treatment=1 and one with treatment=0.
        X_val_t1 = np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis=1)
        X_val_t0 = np.concatenate([X_val, np.zeros((X_val.shape[0], 1))], axis=1)
        y1_pred = model.predict(X_val_t1)
        y0_pred = model.predict(X_val_t0)
        ite_pred[val_index] = y1_pred - y0_pred
    return ite_pred


def estimate_ite_svm_cv(X, t, y, cv=3, **model_kwargs):
    """
    ITE estimation using Support Vector Regression with traditional k-fold CV.
    """
    return estimate_ite_traditional_cv(SVR, X, t, y, cv=cv, **model_kwargs)


def estimate_ite_knn_cv(X, t, y, cv=3, **model_kwargs):
    """
    ITE estimation using K-Nearest Neighbors Regressor with traditional k-fold CV.
    """
    return estimate_ite_traditional_cv(KNeighborsRegressor, X, t, y, cv=cv, **model_kwargs)


def estimate_ite_xgb_cv(X, t, y, cv=3, **model_kwargs):
    """
    ITE estimation using XGBoost Regressor with traditional k-fold CV.
    """
    return estimate_ite_traditional_cv(XGBRegressor, X, t, y, cv=cv, **model_kwargs)


##############################################################################
# Evaluate ITE predictions
##############################################################################

def evaluate_ite(y_true, ite_est):
    """
    Model evaluation.
    Computes:
      - ATE estimated as the mean of ite_est.
      - PEHE (precision in estimation of heterogeneous effect) as RMSE.
      - Absolute error in ATE.
    
    For Twins data, if the predicted ITE is computed per individual (length = 2 * len(y_true)),
    we aggregate the predictions per twin-pair by computing:
      aggregated_ite = ite_est[twin1] - ite_est[twin0]
    """
    if len(ite_est) == 2 * len(y_true):
        half = len(y_true)
        ite_est = ite_est[half:] - ite_est[:half]
    ate_est = np.mean(ite_est)
    ate_true = np.mean(y_true)
    ate_abs_error = np.abs(ate_est - ate_true)
    pehe = np.sqrt(mean_squared_error(y_true, ite_est))
    return ate_est, pehe, ate_abs_error


##############################################################################
# Direct model estimation without CV (for reference)
##############################################################################

def estimate_ite_direct(model_class, X_train_pairs, y_train_ite, X_val_pairs, **model_kwargs):
    """
    Noncausal fitting and ITE estimation.
    """
    model = model_class(**model_kwargs)
    model.fit(X_train_pairs, y_train_ite)
    ite_est_val = model.predict(X_val_pairs)
    return ite_est_val


##############################################################################
# T-learner based ITE estimation with 3-fold CV (for causal baselines)
##############################################################################

def estimate_ite_tlearner_cv(model_class, X, t, y, cv=3, **model_kwargs):
    """
    Performs 3-fold cross-validation using a T-learner approach.
    In each fold, the data (X, t, y) are split into training and validation.
    Two separate models are fit:
       - One on the treated observations (t==1) of the training fold.
       - One on the control observations (t==0) of the training fold.
    For each validation fold, potential outcomes are predicted using these two models and
    the estimated ITE is computed as: prediction for treated minus prediction for control.
    The predictions are then aggregated (in the original order) and returned.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    ite_pred = np.empty(X.shape[0])
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        t_train, t_val = t[train_index], t[val_index]
        y_train = y[train_index]
        treated_idx = t_train == 1
        control_idx = t_train == 0
        if np.sum(treated_idx) == 0 or np.sum(control_idx) == 0:
            ite_pred[val_index] = np.mean(y_train[t_train == 1]) - np.mean(y_train[t_train == 0])
            continue
        model_treated = model_class(**model_kwargs)
        model_control = model_class(**model_kwargs)
        model_treated.fit(X_train[treated_idx], y_train[treated_idx])
        model_control.fit(X_train[control_idx], y_train[control_idx])
        y1_pred = model_treated.predict(X_val)
        y0_pred = model_control.predict(X_val)
        ite_pred[val_index] = y1_pred - y0_pred
    return ite_pred


def estimate_ite_interaction_cv(X, t, y, cv=3, **model_kwargs):
    """
    ITE estimation using Interacted Linear Regression (T-learner on interaction features)
    with 3-fold CV. Interaction features are generated using PolynomialFeatures.
    """
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_inter = poly.fit_transform(X)
    return estimate_ite_tlearner_cv(LinearRegression, X_inter, t, y, cv=cv, **model_kwargs)
