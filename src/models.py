"""Level-1 model zoo. XGBoost and RealMLP are optional (GPU recommended)."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def hgbr(cat_idx):
    return HistGradientBoostingRegressor(categorical_features=cat_idx, max_iter=600,
                                         learning_rate=0.04, l2_regularization=1.0,
                                         random_state=42)


def ridge():
    return Ridge(alpha=10.0)


def linear_svr():
    return LinearSVR(C=0.1, epsilon=0.05, dual=False,
                     loss="squared_epsilon_insensitive", random_state=42)


def xgboost(device):
    from xgboost import XGBRegressor
    return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                        n_estimators=15_000, learning_rate=0.01, max_depth=4,
                        min_child_weight=10, subsample=0.9, colsample_bytree=1.0,
                        reg_alpha=0.0, reg_lambda=4.0, gamma=0.0, random_state=42,
                        n_jobs=-1, early_stopping_rounds=50, device=device)


def realmlp(device):
    from pytabkit import RealMLP_TD_Regressor
    return RealMLP_TD_Regressor(
        random_state=42, device=device, verbosity=0, val_metric_name="rmse",
        n_epochs=60, batch_size=16384, n_ens=1, use_early_stopping=True,
        early_stopping_additive_patience=20, early_stopping_multiplicative_patience=1,
        act="mish", embedding_size=8, first_layer_lr_factor=0.596, hidden_sizes="rectangular",
        hidden_width=384, lr=0.03, ls_eps=0.0114, ls_eps_sched="coslog4",
        max_one_hot_cat_size=18, n_hidden_layers=3, p_drop=0.073, p_drop_sched="flat_cos",
        plr_hidden_1=16, plr_hidden_2=8, plr_lr_factor=0.115, plr_sigma=2.33,
        scale_lr_factor=2.24, sq_mom=1.0 - 0.0118, wd=0.0236)
