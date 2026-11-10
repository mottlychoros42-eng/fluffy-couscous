"""
Small system settings kept in the database (table `app_settings`, created automatically).

    safety_alerts_enabled      true / false   Safety Limit Alerts. Only the predefined-limit alerts follow this switch.
                                              ML alerts and sensor-offline alerts ignore it completely.
    telemetry_retention_days   0 = keep forever, else days of live telemetry to keep (only the `telemetry` table)
    telemetry_dedupe_enabled   true / false   skip live readings that did not change (see telemetry_store.py)
    telemetry_heartbeat_s      seconds        longest gap between two stored live readings while nothing changes

If the database cannot be read, the defaults are used. The default for the safety switch is ON, so a
database hiccup can never silently turn safety alerts off.
"""
import threading
import time

from fastapi import APIRouter, HTTPException, Request

DEFAULTS = {
    "safety_alerts_enabled": "true",
    "telemetry_retention_days": "0",
    "telemetry_dedupe_enabled": "true",
    "telemetry_heartbeat_s": "30",
}

DDL = """CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)"""

_BOOL_KEYS = {"safety_alerts_enabled", "telemetry_dedupe_enabled"}
_INT_RANGES = {"telemetry_retention_days": (0, 3650), "telemetry_heartbeat_s": (10, 3600)}

_cache = {"t": 0.0, "values": None}
_lock = threading.Lock()
CACHE_TTL = 5.0


def ensure_table(conn):
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()


def invalidate():
    with _lock:
        _cache["t"] = 0.0
        _cache["values"] = None


def _read(conn):
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM app_settings")
    return {r["key"]: r["value"] for r in cur.fetchall()}


def get_all(get_conn, ttl=CACHE_TTL):
    """All settings (defaults filled in), cached for a few seconds. Never raises."""
    now = time.time()
    with _lock:
        if _cache["values"] is not None and now - _cache["t"] < ttl:
            return dict(_cache["values"])
        if _cache["values"] is not None and ttl > 0 and not _cache.get("busy"):
            _cache["busy"] = True                 # stale: answer at once with the last known values, refresh in the background

            def refresh():
                try:
                    get_all(get_conn, ttl=0)
                finally:
                    with _lock:
                        _cache["busy"] = False
            threading.Thread(target=refresh, daemon=True).start()
            return dict(_cache["values"])
    stored = None
    try:
        conn = get_conn()
        try:
            try:
                stored = _read(conn)
            except Exception:
                conn.rollback()
                ensure_table(conn)          # first run: create the table, then read again
                stored = _read(conn)
        finally:
            conn.close()
    except Exception as e:
        print(f"[SETTINGS] could not read settings, using defaults: {e}", flush=True)
    values = dict(DEFAULTS)
    values.update({k: v for k, v in (stored or {}).items() if k in DEFAULTS})
    if stored is not None:                  # do not cache a failed read: try again next time
        with _lock:
            _cache["values"], _cache["t"] = values, now
    return dict(values)


def get_bool(get_conn, key, default=True):
    v = get_all(get_conn).get(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def get_int(get_conn, key, default=0):
    try:
        return int(float(get_all(get_conn).get(key, default)))
    except (TypeError, ValueError):
        return default


def normalise(key, value):
    """(clean text, None) or (None, error)."""
    if key not in DEFAULTS:
        return None, f"Unknown setting: {key}"
    if key in _BOOL_KEYS:
        if isinstance(value, bool):
            return ("true" if value else "false"), None
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on"):
            return "true", None
        if text in ("false", "0", "no", "off"):
            return "false", None
        return None, f"{key} must be true or false"
    lo, hi = _INT_RANGES[key]
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None, f"{key} must be a whole number"
    if n < lo or n > hi:
        return None, f"{key} must be between {lo} and {hi}"
    return str(n), None


def set_value(conn, key, value):
    clean, error = normalise(key, value)
    if error:
        raise ValueError(error)
    ensure_table(conn)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO app_settings (key, value, updated_at) VALUES (%s, %s, now())
           ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()""",
        (key, clean),
    )
    conn.commit()
    invalidate()
    return clean


def public_view(values):
    """What the web pages get: real booleans and numbers."""
    return {
        "safety_alerts_enabled": str(values["safety_alerts_enabled"]).lower() == "true",
        "telemetry_retention_days": int(values["telemetry_retention_days"]),
        "telemetry_dedupe_enabled": str(values["telemetry_dedupe_enabled"]).lower() == "true",
        "telemetry_heartbeat_s": int(values["telemetry_heartbeat_s"]),
    }


def make_router(get_conn, get_admin):
    router = APIRouter(prefix="/api/settings")

    def require_admin(request: Request):
        if not get_admin(request.cookies.get("admin_session")):
            raise HTTPException(status_code=401, detail="Login required")

    @router.get("")
    def read_settings(request: Request):
        require_admin(request)
        return public_view(get_all(get_conn, ttl=0))

    @router.put("")
    def write_settings(payload: dict, request: Request):
        """Body: any of the settings above, e.g. {"safety_alerts_enabled": false}."""
        require_admin(request)
        payload = payload or {}
        unknown = [k for k in payload if k not in DEFAULTS]
        if unknown or not payload:
            raise HTTPException(status_code=400, detail="Send one or more of: " + ", ".join(DEFAULTS))
        for key, value in payload.items():
            _, error = normalise(key, value)
            if error:
                raise HTTPException(status_code=400, detail=error)
        conn = get_conn()
        try:
            for key, value in payload.items():
                set_value(conn, key, value)
                if key == "safety_alerts_enabled":
                    print(f"[SETTINGS] Safety Limit Alerts {'ENABLED' if str(value).lower() in ('true', '1') else 'DISABLED'} "
                          f"(ML alerts are not affected)", flush=True)
        finally:
            conn.close()
        return public_view(get_all(get_conn, ttl=0))

    return router
