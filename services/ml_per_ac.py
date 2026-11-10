"""
Per-AC anomaly detection (Isolation Forest) for AirBnBrrr.

- One model per `ac_unit`, trained from that unit's rows in `data_gathered`.
- The alert threshold is recalculated on every retrain from the unit's own held-out
  normal data (a percentile of the scores), so it moves with the data.
- Models, thresholds and learning curves are stored in Supabase (tables `ac_models`,
  `ac_threshold_history`, `ac_learning_curve`), because Render's disk is wiped on every redeploy.

The pure logic (clean / train_unit / score_reading / learning_curve / scatter_data) works on
DataFrames and has no database dependency, so it can be tested on its own.
"""
import io
import threading
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# pzem_energy is left out on purpose: it is a cumulative counter that only goes up, so a model
# that sees it would eventually call every new reading "abnormal".
FEATURES = [
    "dht_temp", "dht_humidity", "vibration",
    "ds18b20_temp1", "ds18b20_temp2",
    "pzem_voltage", "pzem_current", "pzem_power",
    "pzem_frequency", "pzem_power_factor",
]
LOG_FEATURES = []                # features with a very long tail would be listed here (seen on a log scale)
KNN_K = 5                        # distance to the 5th nearest normal reading
KNN_SLACK = 1.5                  # distance check must be this many times past the edge of normal before it flags
KNN_REF_MAX = 1500               # normal readings kept in the model for the distance check
MIN_ROWS = 200                   # below this a unit gets no model (no ML alerts until it is trained)
DEFAULT_PCT = 1.0                # threshold = 1st percentile of held-out normal scores


# ----------------------------------------------------------------- pure logic
def prep(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Model features only, forced numeric, long-tail features on a log scale (NaN stays NaN)."""
    feats = list(features or FEATURES)
    out = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in LOG_FEATURES:
        if col in out:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """prep() plus: drop rows that are mostly empty."""
    out = prep(df).dropna(thresh=max(1, len(FEATURES) // 2))
    return out.reset_index(drop=True)


def _knn_dist(z: np.ndarray, ref: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Distance from each row of z to its k-th nearest row in ref (both already in 'normal spread' units)."""
    out = np.empty(len(z))
    for s in range(0, len(z), 200):
        d2 = ((z[s:s + 200, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        kk = min(k, d2.shape[1]) - 1
        out[s:s + 200] = np.sqrt(np.partition(d2, kk, axis=1)[:, kk])
    return out


def bundle_margin(b: dict, filled: pd.DataFrame) -> np.ndarray:
    """How normal each row is for this AC: above 0 = normal, below 0 = unusual.

    Two checks, and the worse one counts:
      * Isolation Forest: unusual combinations and shapes of readings (good with on/off cycles);
      * distance to the nearest normal readings: grows with the SIZE of a change, so it also catches a slow drift
        that the forest scores the same as an edge-of-normal reading.
    """
    margin = b["model"].score_samples(filled) - b["threshold"]
    if "knn_ref" in b:
        cols = list(filled.columns)
        med, scale = pd.Series(b["medians"]).reindex(cols), pd.Series(b["scale"]).reindex(cols)
        z = ((filled - med) / scale).values.astype(float)
        d = _knn_dist(z, np.asarray(b["knn_ref"], dtype=float), b.get("knn_k", KNN_K))
        margin = np.minimum(margin, (b["knn_threshold"] - d) / b["knn_threshold"] * 0.1)
    return margin


def train_unit(df: pd.DataFrame, pct: float = DEFAULT_PCT, n_estimators: int = 200, seed: int = 42):
    """Train one unit's model. `df` must be in time order (oldest first).

    The newest 20% is held out; the threshold is the `pct` percentile of the model's scores on
    that held-out (normal) data. Returns a bundle dict, or None when there is not enough data.
    """
    X = clean(df)
    if len(X) < MIN_ROWS:
        return None
    split = int(len(X) * 0.8)
    fit, val = X.iloc[:split], X.iloc[split:]

    medians = fit.median().fillna(0.0)
    model = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed, n_jobs=1)
    model.fit(fit.fillna(medians))

    # each check is set so that about half of `pct` of held-out normal readings trip it (together about `pct`)
    val_scores = model.score_samples(val.fillna(medians))
    threshold = float(np.percentile(val_scores, pct / 2))

    # robust per-feature spread: explains *which* sensors look odd, and is the unit of the distance check
    mad = (fit - medians).abs().median() * 1.4826
    fallback = fit.std().replace(0, np.nan).fillna(1.0)
    scale = mad.where(mad > 0, fallback).fillna(1.0).clip(lower=1e-6)

    z_fit = ((fit.fillna(medians) - medians) / scale).values.astype(float)
    ref = z_fit if len(z_fit) <= KNN_REF_MAX else z_fit[np.random.default_rng(seed).choice(len(z_fit), KNN_REF_MAX, replace=False)]
    d_val = _knn_dist(((val.fillna(medians) - medians) / scale).values.astype(float), ref)
    knn_threshold = max(float(np.percentile(d_val, 100 - pct / 2)) * KNN_SLACK, 1e-6)

    # "what is normal for this AC": the range that holds 95% of its training readings
    profile = {}
    for col in fit.columns:
        vals = fit[col].dropna()
        if len(vals):
            lo, mid, hi = vals.quantile([0.025, 0.5, 0.975]).tolist()
            profile[col] = {"low": float(lo), "mid": float(mid), "high": float(hi)}

    return {
        "model": model,
        "profile": profile,
        "knn_ref": ref.astype(np.float32),
        "knn_threshold": knn_threshold,
        "knn_k": KNN_K,
        "medians": medians.to_dict(),
        "scale": scale.to_dict(),
        "threshold": threshold,
        "n_rows": int(len(X)),
        "pct": pct,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
    }


IDLE_CURRENT_A = 1.0     # under 1 A the compressor is off between cooling cycles (same rule as the rule engine)
ODD_Z = 3.0              # a reading is "odd" when it is more than 3 normal-spreads from the usual value


def _mode_profile(bundle: dict, running: bool):
    """Usual value and spread of every sensor WHILE RUNNING (or while idle), taken from the normal readings the
    model kept. Comparing a running AC with the all-day median would call every idle reading odd, and the other way round."""
    cache = bundle.setdefault("_mode_cache", {})        # in-memory only: bundles are never saved back after loading
    if running in cache:
        return cache[running]
    prof = None
    feats = list(bundle.get("features", FEATURES))
    ref = np.asarray(bundle["knn_ref"], dtype=float) if "knn_ref" in bundle else None
    if ref is not None and ref.ndim == 2 and ref.shape[1] == len(feats) and "pzem_current" in feats:
        med = np.array([bundle["medians"][f] for f in feats], dtype=float)
        sc = np.array([bundle["scale"][f] for f in feats], dtype=float)
        raw = ref * sc + med                              # back from "normal spread" units to real sensor values
        ci = feats.index("pzem_current")
        sub = raw[raw[:, ci] >= IDLE_CURRENT_A] if running else raw[raw[:, ci] < IDLE_CURRENT_A]
        if len(sub) >= 30:
            mid = np.median(sub, axis=0)
            mad = np.median(np.abs(sub - mid), axis=0) * 1.4826
            prof = {f: (float(mid[i]), float(mad[i])) for i, f in enumerate(feats)}
    cache[running] = prof
    return prof


def deviations(bundle: dict, reading: dict):
    """(running?, [how far each measured sensor is from this AC's usual value, biggest first]).

    Each item: key, value, mid (usual), low/high (usual range = usual +- 3 spreads), z (how many spreads away), direction.
    Sensors that sent nothing (offline) are skipped: the model had to guess them, so they say nothing.
    """
    feats = list(bundle.get("features", FEATURES))
    try:
        cur = reading.get("pzem_current")
        running = True if cur is None else float(cur) >= IDLE_CURRENT_A
    except (TypeError, ValueError):
        running = True
    prof = _mode_profile(bundle, running)
    out = []
    for f in feats:
        try:
            x = float(reading.get(f))
        except (TypeError, ValueError):
            continue
        if x != x:
            continue
        if prof and f in prof:
            mid, mad = prof[f]
            spread = max(float(bundle["scale"][f]), mad)
        else:
            mid, spread = float(bundle["medians"][f]), float(bundle["scale"][f])
        spread = max(spread, 1e-6)
        z = abs(x - mid) / spread
        out.append({"key": f, "value": x, "mid": mid, "low": mid - ODD_Z * spread, "high": mid + ODD_Z * spread,
                    "z": float(z), "direction": "above" if x > mid else "below"})
    out.sort(key=lambda d: -d["z"])
    return running, out


def score_reading(bundle: dict, reading: dict) -> dict:
    """Score one live reading (dict of sensor values) against a unit's bundle."""
    feats = bundle.get("features", FEATURES)
    row = prep(pd.DataFrame([{f: reading.get(f) for f in feats}]), feats)
    med = pd.Series(bundle["medians"])
    filled = row.fillna(med)
    score = float(bundle["model"].score_samples(filled)[0])
    margin = float(bundle_margin(bundle, filled)[0])

    # which of the two checks says "unusual", and how far past the edge of normal the reading is
    forest_margin = score - bundle["threshold"]
    distance_ratio, distance_margin = None, None
    if "knn_ref" in bundle:
        cols = list(filled.columns)
        z_row = ((filled - med.reindex(cols)) / pd.Series(bundle["scale"]).reindex(cols)).values.astype(float)
        d = float(_knn_dist(z_row, np.asarray(bundle["knn_ref"], dtype=float), bundle.get("knn_k", KNN_K))[0])
        distance_ratio = d / bundle["knn_threshold"]
        distance_margin = (bundle["knn_threshold"] - d) / bundle["knn_threshold"] * 0.1
    checks = [n for n, m in (("forest", forest_margin), ("distance", distance_margin)) if m is not None and m < 0]

    running, devs = deviations(bundle, reading)
    odd = [d for d in devs if d["z"] > ODD_Z]
    top = [(d["key"], round(d["z"], 1)) for d in odd[:3]]
    return {
        "score": score,
        "threshold": bundle["threshold"],
        "margin": margin,                       # above 0 = normal for this AC
        "is_anomaly": margin < 0,
        "top_features": top,
        "forest_margin": float(forest_margin),
        "distance_ratio": None if distance_ratio is None else round(float(distance_ratio), 2),
        "checks": checks,                       # which checks said "unusual": "forest" and/or "distance"
        "operating_mode": "running" if running else "idle",
        "deviations": [{**d, **{k: round(d[k], 3) for k in ("value", "mid", "low", "high")}, "z": round(d["z"], 1)} for d in odd[:8]],
    }


def learning_curve(df: pd.DataFrame, steps: int = 10, pct: float = DEFAULT_PCT) -> pd.DataFrame:
    """Retrain on a growing slice of history and measure on a fixed hold-out.

    Columns: rows, threshold, holdout_flag_rate (share of hold-out normal data that would raise
    an alert) and holdout_mean_score.
    """
    cols = ["rows", "threshold", "holdout_flag_rate", "holdout_mean_score"]
    X = clean(df)
    hold_start = int(len(X) * 0.8)
    pool, hold = X.iloc[:hold_start], X.iloc[hold_start:]
    if len(pool) < MIN_ROWS or hold.empty:
        return pd.DataFrame(columns=cols)

    sizes = sorted(set(np.linspace(MIN_ROWS, len(pool), steps).astype(int)))
    rows = []
    for k in sizes:
        b = train_unit(pool.iloc[:k], pct=pct)
        if b is None:
            continue
        hs = bundle_margin(b, hold.fillna(pd.Series(b["medians"])))
        rows.append({
            "rows": k,
            "threshold": b["threshold"],
            "holdout_flag_rate": float((hs < 0).mean()),
            "holdout_mean_score": float(hs.mean()),
        })
    return pd.DataFrame(rows, columns=cols)


def scatter_data(bundles: dict, frames: dict) -> dict:
    """2D picture of the models for the Isolation Forest page.

    frames = {ac_unit: DataFrame(id, timestamp, ac_unit + FEATURES), oldest first}. Every unit is
    scored by ITS OWN model; all points are projected onto the same 2D plane (PCA).
    Same JSON shape the page already used: points[] + stats{}.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    parts = []
    for unit, df in frames.items():
        b = bundles.get(unit)
        if b is None or df is None or len(df) == 0:
            continue
        df = df.reset_index(drop=True)
        filled = prep(df, b.get("features")).fillna(pd.Series(b["medians"]))
        margin = bundle_margin(b, filled)                              # < 0 means abnormal
        parts.append({"unit": unit, "df": df, "X": filled, "margin": margin, "cutoff": int(len(df) * 0.8), "pct": b["pct"]})

    empty = {"points": [], "stats": {"total": 0, "regular": 0, "abnormal": 0,
                                     "contamination": DEFAULT_PCT / 100, "pca_variance": [0, 0],
                                     "source_table": "data_gathered"}}
    if not parts or sum(len(p["df"]) for p in parts) < 3:
        return empty

    pooled = pd.concat([p["X"] for p in parts], ignore_index=True)
    coords = PCA(n_components=2, random_state=42)
    xy = coords.fit_transform(StandardScaler().fit_transform(pooled))

    points, regular, abnormal, off = [], 0, 0, 0
    for p in parts:
        df = p["df"]
        for i in range(len(df)):
            is_anom = bool(p["margin"][i] < 0)
            if is_anom:
                cat, abnormal = "abnormal", abnormal + 1
            elif i < p["cutoff"]:
                cat, regular = "training", regular + 1
            else:
                cat, regular = "regular", regular + 1

            def val(col):
                v = pd.to_numeric(df.at[i, col], errors="coerce") if col in df.columns else np.nan
                return float(v) if pd.notnull(v) else 0

            points.append({
                "id": int(df.at[i, "id"]) if "id" in df.columns and pd.notnull(df.at[i, "id"]) else i,
                "timestamp": str(df.at[i, "timestamp"]) if "timestamp" in df.columns and pd.notnull(df.at[i, "timestamp"]) else "",
                "ac_unit": p["unit"],
                "x": round(float(xy[off + i, 0]), 3),
                "y": round(float(xy[off + i, 1]), 3),
                "category": cat,
                "is_anomaly": is_anom,
                "score": round(float(p["margin"][i]), 4),
                "pzem_power": val("pzem_power"),
                "dht_temp": val("dht_temp"),
                "pzem_voltage": val("pzem_voltage"),
                "vibration": val("vibration"),
            })
        off += len(df)

    return {"points": points, "stats": {
        "total": len(points), "regular": regular, "abnormal": abnormal,
        "contamination": parts[0]["pct"] / 100,
        "pca_variance": [round(float(v), 4) for v in coords.explained_variance_ratio_],
        "source_table": "data_gathered",
    }}


UNITS = {
    "dht_temp": "°C", "dht_humidity": "%", "vibration": "", "ds18b20_temp1": "°C", "ds18b20_temp2": "°C",
    "pzem_voltage": "V", "pzem_current": "A", "pzem_power": "W", "pzem_frequency": "Hz", "pzem_power_factor": "",
}


def bucket_health(df: pd.DataFrame, max_points: int = 240) -> list:
    """Average the logged health margins into at most `max_points` points for the chart."""
    if df is None or len(df) == 0:
        return []
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["flagged"] = d["flagged"].astype(int)
    d["low"] = d["margin"]                      # the worst point in each bucket, so short dips are not averaged away
    span = d["ts"].max() - d["ts"].min()
    bucket = max(span / max_points, pd.Timedelta(minutes=1))
    g = d.set_index("ts").resample(bucket).agg({"margin": "mean", "low": "min", "flagged": "max"}).dropna()
    return [{"t": ts.isoformat(), "margin": float(r["margin"]), "low": float(r["low"]), "flagged": bool(r["flagged"])} for ts, r in g.iterrows()]


def trend(df: pd.DataFrame, min_days: int = 5, min_per_day: int = 30) -> dict:
    """Is this AC drifting away from its normal? Needs several days of logged health.

    `margin` is the score minus the alert line, so above 0 is normal. A steady downward slope over
    the days is an early warning; the estimate assumes the same drift continues, nothing more.
    """
    if df is None or len(df) == 0:
        return {"status": "not_enough", "days_with_data": 0, "needed": min_days}
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["day"] = d["ts"].dt.floor("D")
    g = d.groupby("day")["margin"].agg(["mean", "count"])
    g = g[g["count"] >= min_per_day]
    if len(g) < min_days:
        return {"status": "not_enough", "days_with_data": int(len(g)), "needed": min_days}
    x = np.array([(day - g.index[0]).days for day in g.index], dtype=float)
    y = g["mean"].values
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    current = float(y[-3:].mean())
    out = {"status": "stable", "days_with_data": int(len(g)), "current_margin": current,
           "slope_per_day": float(slope), "r2": float(r2)}
    if current < 0:
        out["status"] = "outside"
    elif slope < 0 and r2 >= 0.3 and (-slope * 14) > float(resid.std() or 0):
        days = current / (-slope)
        if days <= 120:
            out["status"], out["days_to_alert_line"] = "declining", float(days)
    elif slope > 0 and r2 >= 0.3:
        out["status"] = "improving"
    return out


def plot_learning(curves: dict, path: str, pct: float = DEFAULT_PCT):
    """Optional PNG of the learning curves (needs matplotlib: pip install matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for unit, c in curves.items():
        if c.empty:
            continue
        a1.plot(c["rows"], c["threshold"], marker="o", label=unit)
        a2.plot(c["rows"], c["holdout_flag_rate"] * 100, marker="o", label=unit)
    a1.set_title("Alert threshold as the model sees more data")
    a1.set_xlabel("training rows"); a1.set_ylabel("threshold (anomaly score)")
    a2.axhline(pct, ls="--", color="gray", label=f"target {pct}%")
    a2.set_title("False-alarm rate on unseen normal data")
    a2.set_xlabel("training rows"); a2.set_ylabel("% flagged")
    for a in (a1, a2):
        a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------ database layer
DDL = [
    """CREATE TABLE IF NOT EXISTS ac_models (
        ac_unit TEXT PRIMARY KEY,
        model BYTEA NOT NULL,
        threshold DOUBLE PRECISION NOT NULL,
        n_rows INTEGER NOT NULL,
        trained_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    """CREATE TABLE IF NOT EXISTS ac_threshold_history (
        id SERIAL PRIMARY KEY,
        ac_unit TEXT NOT NULL,
        trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        threshold DOUBLE PRECISION NOT NULL,
        n_rows INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS ac_health (
        id SERIAL PRIMARY KEY,
        ac_unit TEXT NOT NULL,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        margin DOUBLE PRECISION NOT NULL,
        flagged BOOLEAN NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS ac_health_unit_ts ON ac_health (ac_unit, ts)",
    """CREATE TABLE IF NOT EXISTS ac_learning_curve (
        id SERIAL PRIMARY KEY,
        ac_unit TEXT NOT NULL,
        trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        n_rows INTEGER NOT NULL,
        threshold DOUBLE PRECISION NOT NULL,
        holdout_flag_rate DOUBLE PRECISION NOT NULL,
        pct DOUBLE PRECISION NOT NULL
    )""",
]


def ensure_tables(conn):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    conn.commit()


def fetch_units(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT ac_unit FROM data_gathered WHERE ac_unit IS NOT NULL ORDER BY ac_unit")
    return [r["ac_unit"] for r in cur.fetchall()]


def fetch_unit_df(conn, unit: str, limit: int = 20000) -> pd.DataFrame:
    """Newest `limit` rows for a unit, returned oldest-first (features only)."""
    cols = ", ".join(FEATURES)
    cur = conn.cursor()
    cur.execute(
        f"SELECT {cols} FROM data_gathered WHERE ac_unit = %s ORDER BY id DESC LIMIT %s",
        (unit, limit),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True) if rows else pd.DataFrame(columns=FEATURES)


def fetch_unit_rows(conn, unit: str, limit: int = 500) -> pd.DataFrame:
    """Newest `limit` rows for a unit incl. id and timestamp, oldest-first (for the scatter page)."""
    cols = ", ".join(FEATURES)
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, timestamp, ac_unit, {cols} FROM data_gathered WHERE ac_unit = %s ORDER BY id DESC LIMIT %s",
        (unit, limit),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True) if rows else pd.DataFrame()


def save_bundle(conn, unit: str, bundle: dict):
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO ac_models (ac_unit, model, threshold, n_rows, trained_at)
           VALUES (%s, %s, %s, %s, now())
           ON CONFLICT (ac_unit) DO UPDATE SET model = EXCLUDED.model,
             threshold = EXCLUDED.threshold, n_rows = EXCLUDED.n_rows, trained_at = now()""",
        (unit, buf.getvalue(), bundle["threshold"], bundle["n_rows"]),
    )
    cur.execute(
        "INSERT INTO ac_threshold_history (ac_unit, threshold, n_rows) VALUES (%s, %s, %s)",
        (unit, bundle["threshold"], bundle["n_rows"]),
    )
    conn.commit()


def save_curve(conn, unit: str, curve: pd.DataFrame, pct: float):
    """Replace the stored learning curve of one unit (used by the learning page)."""
    cur = conn.cursor()
    cur.execute("DELETE FROM ac_learning_curve WHERE ac_unit = %s", (unit,))
    for r in curve.itertuples(index=False):
        cur.execute(
            """INSERT INTO ac_learning_curve (ac_unit, n_rows, threshold, holdout_flag_rate, pct)
               VALUES (%s, %s, %s, %s, %s)""",
            (unit, int(r.rows), float(r.threshold), float(r.holdout_flag_rate), float(pct)),
        )
    conn.commit()


def fetch_curve(conn, unit: str) -> list:
    cur = conn.cursor()
    cur.execute(
        """SELECT n_rows AS rows, threshold, holdout_flag_rate, pct FROM ac_learning_curve
           WHERE ac_unit = %s ORDER BY n_rows ASC""",
        (unit,),
    )
    return [dict(r) for r in cur.fetchall()]


def log_health(conn, unit: str, margin: float, flagged: bool):
    """One health point (score minus alert line). The caller limits this to about one per minute per AC."""
    cur = conn.cursor()
    cur.execute("INSERT INTO ac_health (ac_unit, margin, flagged) VALUES (%s, %s, %s)", (unit, float(margin), bool(flagged)))
    conn.commit()


def prune_health(conn, days: int = 45):
    from datetime import timedelta
    cur = conn.cursor()
    cur.execute("DELETE FROM ac_health WHERE ts < %s", (datetime.now(timezone.utc) - timedelta(days=days),))
    conn.commit()


def fetch_health(conn, unit: str, days: int = 30) -> pd.DataFrame:
    from datetime import timedelta
    cur = conn.cursor()
    cur.execute("SELECT ts, margin, flagged FROM ac_health WHERE ac_unit = %s AND ts >= %s ORDER BY ts ASC",
                (unit, datetime.now(timezone.utc) - timedelta(days=days)))
    rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ts", "margin", "flagged"])


def load_bundles(conn, errors: dict = None) -> dict:
    """Every stored model, by ac_unit. A model that cannot be read (damaged, or saved by an incompatible
    library version) is skipped and noted in `errors` instead of taking all the other models down with it."""
    cur = conn.cursor()
    cur.execute("SELECT ac_unit, model FROM ac_models")
    out = {}
    for r in cur.fetchall():
        try:
            out[r["ac_unit"]] = joblib.load(io.BytesIO(bytes(r["model"])))
        except Exception as e:
            if errors is not None:
                errors[r["ac_unit"]] = str(e)
            print(f"[ML] model of {r['ac_unit']} could not be loaded, it must be retrained: {e}", flush=True)
    return out


def scatter_from_db(conn, bundles: dict, unit=None, limit: int = 500) -> dict:
    """Scatter data for one unit, or for every unit that has a model when `unit` is empty."""
    units = [unit] if unit else sorted(bundles)
    per_unit = limit if unit else max(50, limit // max(1, len(units)))
    frames = {u: fetch_unit_rows(conn, u, per_unit) for u in units if u in bundles}
    return scatter_data(bundles, frames)


# ------------------------------------------------------- live scoring (backend)
class LiveDetector:
    """Loads the saved per-AC models and scores incoming readings.

    Reloads from Supabase every `refresh_seconds`, so a retrain or an AC deletion is picked up
    without redeploying.
    """

    def __init__(self, get_conn, refresh_seconds: int = 900):
        self._get_conn = get_conn
        self._refresh_seconds = refresh_seconds
        self._bundles = {}
        self._errors = {}                # ac_unit -> why its stored model could not be loaded
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def drop(self, unit):
        """Forget one AC's model right now (its training data was deleted) and reload on the next use.
        Without this the old model would keep scoring until the next refresh."""
        with self._lock:
            self._bundles.pop(unit, None)
            self._errors.pop(unit, None)
            self._loaded_at = 0.0

    def load_error(self, unit):
        self._maybe_refresh()
        return self._errors.get(unit)

    def bundle_info(self, unit):
        """Small description of one AC's loaded model, or None."""
        self._maybe_refresh()
        b = self._bundles.get(unit)
        return None if b is None else {"n_rows": b.get("n_rows"), "trained_at": b.get("trained_at"), "pct": b.get("pct")}

    def invalidate(self):
        """Force a reload from Supabase on the next use (after train/delete)."""
        with self._lock:
            self._loaded_at = 0.0

    def _maybe_refresh(self):
        if time.time() - self._loaded_at < self._refresh_seconds:
            return
        with self._lock:
            if time.time() - self._loaded_at < self._refresh_seconds:
                return
            try:
                conn = self._get_conn()
                try:
                    errors = {}
                    self._bundles = load_bundles(conn, errors)
                    self._errors = errors
                finally:
                    conn.close()
            except Exception as e:  # keep the old models if Supabase hiccups (or tables not created yet)
                print(f"[ML] model refresh failed: {e}", flush=True)
            self._loaded_at = time.time()

    def get_bundles(self) -> dict:
        self._maybe_refresh()
        return dict(self._bundles)

    def check(self, unit, reading: dict):
        """Return a result dict, or None when this unit has no (usable) model.
        None means "the model has no opinion": callers must NOT replace it with threshold rules and call that ML."""
        self._maybe_refresh()
        bundle = self._bundles.get(unit)
        if bundle is None:
            return None
        result = score_reading(bundle, reading)
        result["trained_at"] = bundle.get("trained_at")      # which model said this (shown on the dashboard)
        result["n_rows"] = bundle.get("n_rows")
        return result
