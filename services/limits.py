"""
Per-AC safety limits, editable on the Manage ACs page.

Every AC starts with the DEFAULT_LIMITS below (the 1.5 HP window AC the project began with). Changing
a limit for one AC is stored in Supabase (table `ac_limits`) and used by:
  * the rule engine (what counts as a problem and what the alert says),
  * the instant safety alerts (current, voltage, power),
  * the "readings when the alert fired" panel and the guideline card.
Nothing else needs editing when a different AC is added: set its limits, or train its model.
"""
import json
import math
import threading
import time

DEFAULT_LIMITS = {
    "power":       {"min": 1650, "max": 1860},
    "voltage":     {"min": 225,  "max": 241},
    "current":     {"min": 7.6,  "max": 8.8},
    "output_temp": {"min": 7,    "max": 25},
    "humidity":    {"min": 30,   "max": 100},   # LOW humidity only: the high side is not an alert (kept at 100, hidden)
    "outlet_temp": {"min": 50,   "max": 61},
    "inlet_temp":  {"min": 8,    "max": 17},
    "vibration":   {"min": 10,   "max": 90},
}

# What the web form shows. `use` says which side matters for that sensor ("both", "min" = only the low side,
# "max" = only the high side). Humidity is "min": high humidity is normal for cooled air and is not an alert.
LIMIT_META = [
    {"key": "power",       "label": "Power usage",            "unit": "W",  "use": "both", "instant": "max", "step": 10},
    {"key": "voltage",     "label": "Supply voltage",         "unit": "V",  "use": "both", "instant": "max", "step": 1},
    {"key": "current",     "label": "Operating current",      "unit": "A",  "use": "both", "instant": "max", "step": 0.1},
    {"key": "output_temp", "label": "Output temperature",     "unit": "°C", "use": "both", "step": 1},
    {"key": "humidity",    "label": "Output humidity",        "unit": "%",  "use": "min",  "step": 1,
     "min_label": "Alert when below"},
    {"key": "outlet_temp", "label": "Outlet compressor temp", "unit": "°C", "use": "both", "step": 1},
    {"key": "inlet_temp",  "label": "Inlet compressor temp",  "unit": "°C", "use": "both", "step": 1},
    {"key": "vibration",   "label": "Compressor vibration",   "unit": "Hz", "use": "both", "step": 1,
     "min_label": "Not running below"},
]


def merged(overrides=None):
    """DEFAULT_LIMITS with the saved changes of one AC applied on top (floats, unknown keys ignored)."""
    out = {k: dict(v) for k, v in DEFAULT_LIMITS.items()}
    for key, pair in (overrides or {}).items():
        if key in out and isinstance(pair, dict):
            for side in ("min", "max"):
                if side in pair:
                    try:
                        out[key][side] = float(pair[side])
                    except (TypeError, ValueError):
                        pass
    # Before v3.0 humidity was a HIGH-only limit and the form always saved its low side as 0, so a saved 0 was never a
    # choice: use the default low-humidity limit instead (otherwise low humidity could never alert for existing ACs).
    if out["humidity"]["min"] <= 0:
        out["humidity"]["min"] = DEFAULT_LIMITS["humidity"]["min"]
    out["humidity"]["max"] = max(out["humidity"]["max"], 100.0)    # the high side is not an alert any more
    return out


def validate(data):
    """Return (clean limits, error text). `data` is {sensor: {min, max}}; both sides are required."""
    if not isinstance(data, dict):
        return None, "Limits must be an object"
    clean = {}
    for meta in LIMIT_META:
        key = meta["key"]
        pair = data.get(key)
        if not isinstance(pair, dict):
            return None, f"Missing limits for {meta['label']}"
        vals = {}
        for side in ("min", "max"):
            try:
                v = float(pair.get(side))
            except (TypeError, ValueError):
                return None, f"{meta['label']}: {side} must be a number"
            if not math.isfinite(v) or v < 0 or v > 100000:
                return None, f"{meta['label']}: {side} must be between 0 and 100000"
            vals[side] = v
        if vals["min"] >= vals["max"]:
            return None, f"{meta['label']}: the low value must be smaller than the high value"
        clean[key] = vals
    return clean, None


SUGGEST_COLUMNS = {
    "power": "pzem_power", "voltage": "pzem_voltage", "current": "pzem_current", "output_temp": "dht_temp",
    "humidity": "dht_humidity", "outlet_temp": "ds18b20_temp1", "inlet_temp": "ds18b20_temp2", "vibration": "vibration",
}


def suggest(df, idle_current=1.0, min_rows=100):
    """Limits that fit ONE AC, from its own readings while the compressor was running.

    Takes the range that holds 99% of those readings and adds 10% on each side. Returns (limits, rows used),
    or (None, rows) when there are not enough running readings. The caller shows it for review: nothing is saved.
    """
    import pandas as pd
    d = df.apply(pd.to_numeric, errors="coerce")
    run = d[d["pzem_current"] >= idle_current]
    if len(run) < min_rows:
        return None, int(len(run))
    out = {}
    for key, col in SUGGEST_COLUMNS.items():
        v = run[col].dropna()
        if len(v) < min_rows:
            out[key] = dict(DEFAULT_LIMITS[key])
            continue
        lo, hi = (float(x) for x in v.quantile([0.005, 0.995]))
        pad = max((hi - lo) * 0.10, abs(hi) * 0.01, 0.1)
        low, high = max(lo - pad, 0.0), hi + pad
        if key == "humidity":
            high = 100.0            # only LOW humidity is an alert: the high side stays out of the way
        out[key] = {"min": round(low, 1), "max": round(min(high, 100.0) if key == "humidity" else high, 1)}
        if out[key]["min"] >= out[key]["max"]:
            out[key] = dict(DEFAULT_LIMITS[key])
    # the instant safety limits are never tightened by a suggestion (a steady AC would then alert on small wobbles);
    # a bigger AC can loosen them, and you can always tighten them by hand
    for key in ("current", "voltage", "power"):
        out[key]["max"] = max(out[key]["max"], float(DEFAULT_LIMITS[key]["max"]))
    return out, int(len(run))


# ------------------------------------------------------------------ database
def ensure_table(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS ac_limits (
        ac_unit TEXT PRIMARY KEY,
        limits TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""")
    conn.commit()


def load_custom(conn, unit):
    """The saved limits of one AC, or None when it uses the defaults."""
    cur = conn.cursor()
    cur.execute("SELECT limits FROM ac_limits WHERE ac_unit = %s", (unit,))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row["limits"])
    except (TypeError, ValueError):
        return None


def save(conn, unit, limits):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO ac_limits (ac_unit, limits, updated_at) VALUES (%s, %s, now())
           ON CONFLICT (ac_unit) DO UPDATE SET limits = EXCLUDED.limits, updated_at = now()""",
        (unit, json.dumps(limits)),
    )
    conn.commit()
    invalidate(unit)


def reset(conn, unit):
    cur = conn.cursor()
    cur.execute("DELETE FROM ac_limits WHERE ac_unit = %s", (unit,))
    conn.commit()
    invalidate(unit)


# ------------------------------------------------------- cached lookup (alerts)
_cache = {}
_lock = threading.Lock()


def invalidate(unit=None):
    with _lock:
        if unit is None:
            _cache.clear()
        else:
            _cache.pop(unit, None)


_refreshing = set()


def _load(get_conn, unit):
    try:
        conn = get_conn()
        try:
            custom = load_custom(conn, unit)
        finally:
            conn.close()
        return merged(custom)
    except Exception:
        return None


def get(get_conn, unit, ttl=30):
    """Limits of one AC for the alert code. Falls back to the defaults if the database is unavailable.

    Never makes the alert wait for the database once it has been read once: an out-of-date copy is returned at once
    and refreshed in the background (the alert decision must not depend on how fast Supabase answers).
    """
    now = time.time()
    with _lock:
        hit = _cache.get(unit)
        if hit and now - hit[0] < ttl:
            return hit[1]
        if hit:                                   # stale: use it, refresh in the background (once)
            if unit not in _refreshing:
                _refreshing.add(unit)

                def refresh():
                    value = _load(get_conn, unit)
                    with _lock:
                        if value is not None:
                            _cache[unit] = (time.time(), value)
                        _refreshing.discard(unit)
                threading.Thread(target=refresh, daemon=True).start()
            return hit[1]
    value = _load(get_conn, unit) or merged(None)   # first time only: table not created yet, or the database is down
    with _lock:
        _cache[unit] = (now, value)
    return value