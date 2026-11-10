"""
Live telemetry housekeeping (v3.0).

What the `telemetry` table is for (checked in the code, see documentation/telemetry_review.md):
  * the dashboard's latest values, the real-time graphs, the detailed / single graphs, the daily energy figure.
  * NOT the ML model (it trains on `data_gathered` only and scores one reading at a time),
  * NOT the alerts (they are decided from the incoming reading itself, before anything is stored).

So the live table can be thinned and cleaned without touching the models, the training data or the alerts.

1. Duplicate skipping  - a live reading that is practically the same as the last STORED one (inside a small
   band per sensor) is not stored again, but one is always stored at least every `telemetry_heartbeat_s`
   seconds and on the first reading of a new day (the daily energy figure starts from it). The reading is
   still scored by the model and the safety limits BEFORE this check, so alerting never sees a gap.
   Training data (`data_gathered`) is never skipped: every reading is kept at full detail.
2. Retention           - optional: delete live readings older than N days. Only the `telemetry` table is
   touched: never `data_gathered`, models, thresholds, limits or alerts.
"""
import threading
import time

from fastapi import APIRouter, HTTPException, Request

# How much a sensor may wobble before a reading counts as "different". Energy is left out: it only counts up.
DEDUPE_BANDS = {
    "dht_temp": 0.3, "dht_humidity": 1.5, "vibration": 2.0,
    "ds18b20_temp1": 0.3, "ds18b20_temp2": 0.3,
    "pzem_voltage": 1.0, "pzem_current": 0.05, "pzem_power": 5.0,
    "pzem_frequency": 0.1, "pzem_power_factor": 0.02,
}
STATUS_FIELDS = ("ac_status", "ac_thermostat")


def _same(a, b, band):
    if a is None or b is None:
        return a is None and b is None          # a sensor going offline / coming back is always a change
    try:
        return abs(float(a) - float(b)) <= band
    except (TypeError, ValueError):
        return a == b


class Deduper:
    """Remembers the last stored live reading of each device."""

    def __init__(self):
        self._last = {}
        self._lock = threading.Lock()

    def should_store(self, unit, reading, now, day, heartbeat_s, enabled=True):
        """True when this live reading should be written to the `telemetry` table."""
        if not enabled:
            return True
        with self._lock:
            last = self._last.get(unit)
            keep = (
                last is None
                or day != last["day"]
                or now - last["t"] >= heartbeat_s
                or any(not _same(reading.get(k), last["r"].get(k), band) for k, band in DEDUPE_BANDS.items())
                or any(reading.get(k) != last["r"].get(k) for k in STATUS_FIELDS)
            )
            if keep:
                self._last[unit] = {"t": now, "day": day, "r": dict(reading)}
            return keep

    def reset(self):
        with self._lock:
            self._last.clear()


# ----------------------------------------------------------------- database
def prune(conn, days, dry_run=False):
    """Delete (or just count) live readings older than `days`. Touches ONLY the telemetry table."""
    if days is None or int(days) <= 0:
        return 0
    cur = conn.cursor()
    if dry_run:
        cur.execute("SELECT COUNT(*) AS c FROM telemetry WHERE timestamp < now() - (%s * interval '1 day')", (int(days),))
        return int(cur.fetchone()["c"])
    cur.execute("DELETE FROM telemetry WHERE timestamp < now() - (%s * interval '1 day')", (int(days),))
    deleted = cur.rowcount
    conn.commit()
    return int(deleted)


def stats(conn, retention_days=0):
    """Numbers for the data-management card on the Manage ACs page."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c, MIN(timestamp) AS oldest, MAX(timestamp) AS newest FROM telemetry")
    t = cur.fetchone()
    cur.execute("SELECT COUNT(*) AS c FROM data_gathered")
    g = cur.fetchone()
    out = {
        "telemetry_rows": int(t["c"]),
        "telemetry_oldest": t["oldest"].isoformat() if hasattr(t["oldest"], "isoformat") else t["oldest"],
        "telemetry_newest": t["newest"].isoformat() if hasattr(t["newest"], "isoformat") else t["newest"],
        "training_rows": int(g["c"]),
        "telemetry_bytes": None,
        "would_delete": 0,
    }
    try:
        cur.execute("SELECT pg_total_relation_size('telemetry') AS b")
        out["telemetry_bytes"] = int(cur.fetchone()["b"])
    except Exception:
        conn.rollback()
    if retention_days and int(retention_days) > 0:
        out["would_delete"] = prune(conn, retention_days, dry_run=True)
    return out


def make_router(get_conn, get_admin, get_retention_days):
    router = APIRouter(prefix="/api/telemetry/maintenance")

    def require_admin(request: Request):
        if not get_admin(request.cookies.get("admin_session")):
            raise HTTPException(status_code=401, detail="Login required")

    @router.get("")
    def telemetry_stats(request: Request):
        require_admin(request)
        conn = get_conn()
        try:
            return stats(conn, get_retention_days())
        finally:
            conn.close()

    @router.post("/cleanup")
    def telemetry_cleanup(payload: dict, request: Request):
        """Body: {"days": 30, "dry_run": true}. Deletes live readings older than `days` (telemetry table only)."""
        require_admin(request)
        payload = payload or {}
        try:
            days = int(payload.get("days"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="days must be a whole number")
        if days < 1 or days > 3650:
            raise HTTPException(status_code=400, detail="days must be between 1 and 3650")
        dry = bool(payload.get("dry_run"))
        conn = get_conn()
        try:
            n = prune(conn, days, dry_run=dry)
        finally:
            conn.close()
        return {"dry_run": dry, "days": days, "rows": n, "table": "telemetry"}

    return router
