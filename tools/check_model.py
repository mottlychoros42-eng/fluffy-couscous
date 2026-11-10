#!/usr/bin/env python3
"""
Check what the model would say about YOUR data, using the same code as the live system.

    python tools/check_model.py normal.csv                 trains on the first 80% of normal.csv, checks the newest 20%
    python tools/check_model.py normal.csv abnormal.csv    trains on normal.csv, checks every row of abnormal.csv

Use the "Export CSV" button on Manage ACs for the files (columns like dht_temp, pzem_current, ...). It needs pandas,
numpy and scikit-learn. It prints how many readings are flagged, the severity and condition an alert would get,
and which sensors are odd most often. Nothing is sent or saved.
"""
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import ml_explain, ml_per_ac  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    normal = pd.read_csv(sys.argv[1])
    if len(sys.argv) > 2:
        train, test = normal, pd.read_csv(sys.argv[2])
    else:
        cut = int(len(normal) * 0.8)
        train, test = normal.iloc[:cut], normal.iloc[cut:]
    bundle = ml_per_ac.train_unit(train)
    if bundle is None:
        sys.exit(f"Need at least {ml_per_ac.MIN_ROWS} usable readings to train (got {len(train)}).")
    flagged, sev, cond, odd = 0, Counter(), Counter(), Counter()
    for row in test.to_dict("records"):
        ml = ml_per_ac.score_reading(bundle, row)
        if not ml["is_anomaly"]:
            continue
        flagged += 1
        a = ml_explain.assess(ml, row, "this AC")
        sev[a["severity"]] += 1
        cond[a["condition"]] += 1
        odd.update(o["name"] for o in a["odd"])
    n = max(len(test), 1)
    print(f"Trained on {bundle['n_rows']:,} readings. Checked {len(test):,} readings.")
    print(f"Flagged as abnormal: {flagged:,} ({flagged / n * 100:.1f}%)")
    print("Severity:  ", dict(sev) or "-")
    print("Condition: ", dict(cond) or "-")
    print("Odd sensors (most often first):", ", ".join(f"{k} x{v}" for k, v in odd.most_common(6)) or "-")


if __name__ == "__main__":
    main()
