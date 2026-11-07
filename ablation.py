"""Ablation study: how much does each stage of the pipeline improve CV RMSE?

    python ablation.py --data data/ [--folds 5] [--sample N]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

from src import models
from src.features import BASE_CATS, TARGET, add_numeric_features, build_features

warnings.filterwarnings("ignore")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - p) ** 2)))


def oof_predict(make, X, y, kf):
    oof = np.zeros(len(X))
    for tr, va in kf.split(X):
        Xa, Xb = (X.iloc[tr], X.iloc[va]) if hasattr(X, "iloc") else (X[tr], X[va])
        oof[va] = make().fit(Xa, y.iloc[tr]).predict(Xb)
    return oof


def ordinal(df):
    df = df.copy()
    df[BASE_CATS] = OrdinalEncoder().fit_transform(df[BASE_CATS])
    return df, [df.columns.get_loc(c) for c in BASE_CATS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sample", type=int)
    args = ap.parse_args()

    train = pd.read_csv(Path(args.data) / "train.csv")
    if args.sample:
        train = train.sample(args.sample, random_state=0).reset_index(drop=True)
    y = train[TARGET]
    raw = train.drop(columns=["id", TARGET])
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=123)
    results = {}

    X1, idx1 = ordinal(raw)
    results["1. Baseline: raw features, HGBR"] = rmse(y, oof_predict(lambda: models.hgbr(idx1), X1, y, kf))

    X2, idx2 = ordinal(add_numeric_features(raw))
    results["2. + feature engineering"] = rmse(y, oof_predict(lambda: models.hgbr(idx2), X2, y, kf))

    X3, _, X_lin, _, idx3 = build_features(raw, raw.head(1), y)
    oof_tree = oof_predict(lambda: models.hgbr(idx3), X3, y, kf)
    results["3. + target encoding & pairs"] = rmse(y, oof_tree)

    oof_ridge = oof_predict(models.ridge, X_lin, y, kf)
    oof_svr = oof_predict(models.linear_svr, X_lin, y, kf)
    results["4. Ridge on 15 selected features (parallel branch)"] = rmse(y, oof_ridge)

    S = np.column_stack([oof_tree, oof_ridge, oof_svr])
    blend = Ridge(alpha=1.0).fit(S, y).predict(S)
    results["5. Stacking HGBR + Ridge + SVR"] = rmse(y, np.clip(blend, 19.6, 100.0))

    print("\n| Configuration | CV RMSE |\n|---|---|")
    for k, v in results.items():
        print(f"| {k} | {v:.4f} |")


if __name__ == "__main__":
    main()
