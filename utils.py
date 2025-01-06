import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from econml.dml import DML
from econml.metalearners import XLearner
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression


def estimate_ate_ipw(XZ, t, y):
    """
    ATE estimation via IPW
    """
    scaler = StandardScaler()
    XZ_scaled = scaler.fit_transform(XZ)

    lr = LogisticRegression(max_iter=10000)
    lr.fit(XZ_scaled, t)
    p = lr.predict_proba(XZ_scaled)[:, 1]
    p = np.clip(p, 1e-5, 1 - 1e-5)

    ate_ipw = np.mean(y[t == 1] / p[t == 1]) - np.mean(y[t == 0] / (1 - p[t == 0]))
    return ate_ipw


def estimate_ite_dml(XZ, t, y, XZ_eval):
    """
    ITE estimation via DML
    """
    dml = DML(
        model_y=RandomForestRegressor(random_state=0),
        model_t=RandomForestClassifier(random_state=0),
        model_final=StatsModelsLinearRegression(fit_intercept=False),
        discrete_treatment=True
    )
    dml.fit(y, t, X=XZ)
    ite_est = dml.effect(XZ_eval, T0=0, T1=1)
    return ite_est


def estimate_ite_xlearner(XZ, t, y, XZ_eval):
    """
    ITE estimation via X-Learner
    """
    base_learner = RandomForestRegressor(random_state=0)
    xlearner = XLearner(models=base_learner)
    xlearner.fit(y, t, X=XZ)
    ite_est = xlearner.effect(XZ_eval)
    return ite_est


def evaluate_ite(y_true, ite_est):
    """
    Model evaluation
    """
    ate_est = np.mean(ite_est)
    ate_true = np.mean(y_true)
    ate_abs_error = np.abs(ate_est - ate_true)
    pehe = np.sqrt(mean_squared_error(y_true, ite_est))
    return ate_est, pehe, ate_abs_error


def estimate_ite_direct(model_class, X_train_pairs, y_train_ite, X_val_pairs, **model_kwargs):
    """
    Noncausal fitting and ITE estimation
    """
    model = model_class(**model_kwargs)
    model.fit(X_train_pairs, y_train_ite)
    ite_est_val = model.predict(X_val_pairs)
    return ite_est_val
