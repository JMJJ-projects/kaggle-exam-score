"""Generate a small fake dataset with the competition's schema (for smoke tests / CI)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n, seed, with_target=True):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": r.integers(17, 25, n),
        "gender": r.choice(["male", "female", "other"], n),
        "course": r.choice(["b.sc", "b.tech", "bca", "ba", "bba", "b.com", "diploma"], n),
        "study_hours": r.uniform(0, 8, n).round(2),
        "class_attendance": r.uniform(40, 100, n).round(1),
        "internet_access": r.choice(["yes", "no"], n),
        "sleep_hours": r.uniform(4, 10, n).round(1),
        "sleep_quality": r.choice(["poor", "average", "good"], n),
        "study_method": r.choice(["self-study", "online videos", "group study", "coaching", "mixed"], n),
        "facility_rating": r.choice(["low", "medium", "high"], n),
        "exam_difficulty": r.choice(["easy", "moderate", "hard"], n),
    })
    if with_target:
        score = (6 * df.study_hours + 0.35 * df.class_attendance + 1.5 * df.sleep_hours
                 + 5 * (df.sleep_quality == "good") + 10 * (df.study_method == "coaching")
                 + r.normal(0, 8, n))
        df["exam_score"] = score.clip(19.6, 100).round(1)
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic")
    out.mkdir(exist_ok=True)
    make(4000, 0).to_csv(out / "train.csv", index=False)
    make(1000, 1, with_target=False).assign(id=lambda d: d.id + 4000).to_csv(out / "test.csv", index=False)
    print(f"wrote {out}/train.csv and test.csv")
