"""Train the 2-level stacking ensemble and write a Kaggle submission.

    python train.py --data data/                # full run (GPU recommended)
    python train.py --data data/ --fast         # HGBR + Ridge + SVR only, CPU friendly
    python train.py --data data/ --sample 50000 --folds 3   # quick experiment
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src import models
from src.features import TARGET, build_features

warnings.filterwarnings("ignore")
CLIP = (19.6, 100.0)  # observed range of exam_score in train


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="folder with train.csv and test.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--sample", type=int, help="train on a random subset of N rows")
    ap.add_argument("--fast", action="store_true", help="skip XGBoost and RealMLP")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    if args.sample:
        train = train.sample(args.sample, random_state=0).reset_index(drop=True)

    y = train[TARGET]
    test_ids = test["id"]
    X_tree, Xt_tree, X_lin, Xt_lin, cat_idx = build_features(
        train.drop(columns=["id", TARGET]), test.drop(columns=["id"]), y)
    print(f"{len(train):,} rows | {X_tree.shape[1]} tree features | {X_lin.shape[1]} selected features")

    device = "cuda" if models.has_cuda() else "cpu"
    # name -> (factory, uses selected features?, needs validation set?)
    zoo = {
        "HGBR": (lambda: models.hgbr(cat_idx), False, False),
        "Ridge": (models.ridge, True, False),
        "LinearSVR": (models.linear_svr, True, False),
    }
    if not args.fast:
        for name, lib, make, use_lin in (("XGBoost", "xgboost", lambda: models.xgboost(device), False),
                                         ("RealMLP", "pytabkit", lambda: models.realmlp(device), True)):
            try:
                __import__(lib)
                zoo[name] = (make, use_lin, True)
            except ImportError:
                print(f"{lib} not installed: skipping {name}")
    print("Models:", ", ".join(zoo), f"(device: {device})")

    oof = {k: np.zeros(len(train)) for k in zoo}
    test_pred = {k: np.zeros(len(test)) for k in zoo}
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=123)

    for fold, (tr, va) in enumerate(kf.split(X_tree), 1):
        t0 = time.time()
        for name, (make, use_lin, needs_val) in zoo.items():
            Xa, Xb = (X_lin[tr], X_lin[va]) if use_lin else (X_tree.iloc[tr], X_tree.iloc[va])
            Xtest = Xt_lin if use_lin else Xt_tree
            m = make()
            if name == "XGBoost":
                m.fit(Xa, y.iloc[tr], eval_set=[(Xb, y.iloc[va])], verbose=False)
            elif name == "RealMLP":
                m.fit(Xa, y.iloc[tr], X_val=Xb, y_val=y.iloc[va])
            else:
                m.fit(Xa, y.iloc[tr])
            oof[name][va] = np.ravel(m.predict(Xb))
            test_pred[name] += np.ravel(m.predict(Xtest)) / args.folds
        print(f"fold {fold}/{args.folds} done in {time.time() - t0:.0f}s")

    scores = {name: rmse(y, p) for name, p in oof.items()}

    # Level 2: Ridge blender on out-of-fold predictions.
    S, St = np.column_stack(list(oof.values())), np.column_stack(list(test_pred.values()))
    blender = Ridge(alpha=1.0).fit(S, y)
    scores["Stacking"] = rmse(y, np.clip(blender.predict(S), *CLIP))
    weights = dict(zip(oof, np.round(blender.coef_, 3).tolist()))

    print("\nOut-of-fold RMSE")
    for k, v in scores.items():
        print(f"  {k:<10} {v:.4f}")
    print("Blender weights:", weights)

    pd.DataFrame({"id": test_ids, TARGET: np.clip(blender.predict(St), *CLIP)}) \
        .to_csv(out / "submission.csv", index=False)
    (out / "cv_scores.json").write_text(json.dumps({"rmse": scores, "weights": weights}, indent=2))
    print(f"\nWrote {out / 'submission.csv'}")


if __name__ == "__main__":
    main()
