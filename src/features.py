"""Feature engineering and encoding for the exam-score regression task."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder

TARGET = "exam_score"
NUMS = ["age", "study_hours", "class_attendance", "sleep_hours"]
BASE_CATS = ["gender", "course", "internet_access", "study_method",
             "facility_rating", "exam_difficulty", "sleep_quality"]
# study_hours crossed with categorical context: lets trees and target encoding
# learn a different effect of studying per group.
PAIRS = [("study_hours", "gender"), ("study_hours", "internet_access"),
         ("study_hours", "sleep_quality"), ("study_hours", "study_method")]
PAIR_COLS = [f"{a}_{b}" for a, b in PAIRS]
ALL_CATS = BASE_CATS + PAIR_COLS


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted numeric features (returns a new frame)."""
    df = df.copy()

    # Cyclic signals: the synthetic target has periodic structure in these inputs.
    for p in (12, 14, 20):
        df[f"study_sin_{p}"] = np.sin(2 * np.pi * df["study_hours"] / p)
        df[f"class_sin_{p}"] = np.sin(2 * np.pi * df["class_attendance"] / p)

    for c in NUMS:
        df[f"{c}_sq"] = df[c] ** 2

    df["study_x_attendance"] = df["study_hours"] * df["class_attendance"]
    df["study_sleep_ratio"] = df["study_hours"] / (df["sleep_hours"] + 1e-5)

    # Linear projection fitted by OLS during the data analysis.
    df["proj_ols"] = (5.905 * df["study_hours"] + 0.345 * df["class_attendance"]
                      + 1.423 * df["sleep_hours"] + 4.7819)
    # Domain "points" heuristic: bonuses/penalties for sleep quality, study method, facilities.
    df["proj_points"] = (
        6 * df["study_hours"] + 0.35 * df["class_attendance"] + 1.5 * df["sleep_hours"]
        + 5 * (df["sleep_quality"] == "good") - 5 * (df["sleep_quality"] == "poor")
        + 10 * (df["study_method"] == "coaching") + 5 * (df["study_method"] == "mixed")
        + 2 * (df["study_method"] == "group study") + 1 * (df["study_method"] == "online videos")
        + 4 * (df["facility_rating"] == "high") - 4 * (df["facility_rating"] == "low")
    )
    return df


def add_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for (a, b), name in zip(PAIRS, PAIR_COLS):
        df[name] = df[a].astype(str) + "_" + df[b].astype(str)
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame, y: pd.Series, *,
                   n_select: int = 15, sfs_rows: int = 15_000, seed: int = 42):
    """Full preprocessing used by the final model.

    Returns ``(X_tree, X_tree_test, X_lin, X_lin_test, cat_idx)`` where the *tree*
    matrices hold all engineered + encoded features (for HGBR / XGBoost) and the
    *lin* matrices hold the standardised top-``n_select`` features chosen by forward
    selection (for Ridge / LinearSVR / RealMLP).
    """
    X = add_pairs(add_numeric_features(train))
    Xt = add_pairs(add_numeric_features(test))

    # Cross-fitted target encoding (cv=5) so a row never sees its own target.
    te = TargetEncoder(cv=5, smooth="auto", random_state=seed)
    te_train = te.fit_transform(X[ALL_CATS], y)
    te_test = te.transform(Xt[ALL_CATS])
    for i, col in enumerate(ALL_CATS):
        X[f"TE_{col}"] = te_train[:, i]
        Xt[f"TE_{col}"] = te_test[:, i]

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[ALL_CATS] = oe.fit_transform(X[ALL_CATS])
    Xt[ALL_CATS] = oe.transform(Xt[ALL_CATS])

    scaler = StandardScaler()
    X_scl = scaler.fit_transform(X)
    Xt_scl = scaler.transform(Xt)

    sfs = SequentialFeatureSelector(Ridge(alpha=1.0), n_features_to_select=n_select,
                                    direction="forward", cv=3)
    n = min(sfs_rows, len(X))
    sfs.fit(X_scl[:n], y.iloc[:n])

    cat_idx = [X.columns.get_loc(c) for c in BASE_CATS]
    return X, Xt, sfs.transform(X_scl), sfs.transform(Xt_scl), cat_idx
