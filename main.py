from fastapi import FastAPI, Header, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta
import os
import hmac
import time

LOCAL_TZ = timezone(timedelta(hours=8))
import subprocess
import threading
from database import get_db_connection
from ac_setup import get_ac_setup, update_ac_setup
from web_routes import router as web_router, _read_template, app_version
from services.contacts_api import router as contacts_router
from services.alerts_api import router as alerts_router
from services.traccar_setup_api import router as traccar_setup_router
from services.ai_setup_api import router as ai_setup_router

from services import ml_per_ac, ac_manager, ntfy_notifier, action_guide
from services import limits as limits_store
from services import settings as app_settings
from services import telemetry_store
from services import ml_explain
from services.alert_channels import AlertScheduler, CHANNELS, CHANNEL_LABEL, evaluate_channels, ml_state
from services.anomaly_reciever import process_anomaly
from services.auth import get_admin_from_token, verify_password, generate_session, clear_session

app = FastAPI(title="AirBnBrrr")

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

app.include_router(web_router)
app.include_router(contacts_router)
app.include_router(alerts_router)
app.include_router(traccar_setup_router)
app.include_router(ai_setup_router)

# One Isolation Forest per AC unit, loaded from Supabase (train with train_models.py or the AC manager page)
detector = ml_per_ac.LiveDetector(get_db_connection)
app.include_router(ac_manager.make_router(get_db_connection, get_admin_from_token, detector))
app.include_router(app_settings.make_router(get_db_connection, get_admin_from_token))
app.include_router(telemetry_store.make_router(
    get_db_connection, get_admin_from_token,
    lambda: app_settings.get_int(get_db_connection, "telemetry_retention_days", 0)))

class Telemetry(BaseModel):
	ac_unit: Optional[str] = None
	dust_sensor: Optional[float] = None
	dht_temp: Optional[float] = None
	dht_humidity: Optional[float] = None
	vibration: Optional[float] = None
	ds18b20_temp1: Optional[float] = None
	ds18b20_temp2: Optional[float] = None
	pzem_voltage: Optional[float] = None
	pzem_current: Optional[float] = None
	pzem_power: Optional[float] = None
	pzem_energy: Optional[float] = None
	pzem_frequency: Optional[float] = None
	pzem_power_factor: Optional[float] = None
	ac_status: Optional[str] = None
	ac_thermostat: Optional[str] = None

class ACSetup(BaseModel):
	ac_status: Optional[str] = None
	ac_thermostat: Optional[str] = None
	data_gathering_mode: Optional[str] = None
	data_gathering_unit: Optional[str] = None

class LoginRequest(BaseModel):
	username: str
	password: str

@app.post("/api/login")
def api_login(payload: LoginRequest, response: Response):
	conn = get_db_connection()
	cur = conn.cursor()
	cur.execute("SELECT password FROM admin WHERE username = %s", (payload.username,))
	row = cur.fetchone()
	conn.close()

	if not row:
		raise HTTPException(status_code=401, detail="Invalid username or password")

	hashed = row["password"]
	if not verify_password(payload.password, hashed):
		raise HTTPException(status_code=401, detail="Invalid username or password")

	token, _ = generate_session(payload.username)
	response.set_cookie(
		key="admin_session",
		value=token,
		httponly=True,
		max_age=86400,
		samesite="lax",
	)
	return {"status": "success"}

@app.post("/api/logout")
@app.get("/api/logout")
def api_logout(request: Request, response: Response):
	token = request.cookies.get("admin_session")
	if token:
		clear_session(token)
	response.delete_cookie("admin_session")
	return RedirectResponse(url="/pages/login", status_code=303)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
	token = request.cookies.get("admin_session")
	if not get_admin_from_token(token):
		return RedirectResponse(url="/pages/login", status_code=303)
	return HTMLResponse(_read_template("control_panel.html"))


# --- ALERT CHANNELS (v3.0) -------------------------------------------------------------------------
# Three independent channels judge every reading: "ml" (the trained model of that AC), "safety" (predefined limits,
# switchable) and "sensor" (a sensor is offline). Each has its own timers per AC, so none can delay, replace or
# suppress another. The rules and the timing numbers live in services/alert_channels.py.
scheduler = AlertScheduler()
deduper = telemetry_store.Deduper()

# The newest model verdict of each AC (what the dashboard shows as "ML detection")
_latest_ml = {}
_latest_ml_lock = threading.Lock()


def _remember_ml(unit, ml):
    with _latest_ml_lock:
        _latest_ml[unit] = {"epoch": time.time(), "at": datetime.now(LOCAL_TZ).isoformat(), "ml": dict(ml)}


def _unit_title(unit):
    """'AC1 (device code AC2)' when the dashboard label differs from the device code. From memory only."""
    label = None
    try:
        label = next((u["label"] for u in _units_hot() if u["code"] == unit), None)
    except Exception:
        pass
    if label and label != unit:
        return f"{label} (device code {unit})"
    return label or unit


def _send_alert(unit, channel, is_critical, received_at=None):
    """Send ONE alert of ONE channel for this AC, from that channel's latest snapshot.

    Order matters for instant alerts: ntfy goes out in its own thread at the same moment, the alert is saved and the SMS
    sent next, and only then does the (slow) AI opinion get fetched and attached. Nothing waits for the AI.
    """
    snap = scheduler.snapshot(unit, channel)
    data = dict(snap.get("payload") or {})
    problems = list(snap.get("problems") or [])
    findings = list(snap.get("findings") or [])
    ml = snap.get("ml")
    assessment = snap.get("assessment")
    unit_title = _unit_title(unit)

    print(f"\n[ALERT] {unit}: sending {'INSTANT' if is_critical else 'normal'} {CHANNEL_LABEL[channel].upper()} alert (ntfy + SMS)...", flush=True)

    def push():
        try:
            ok = ntfy_notifier.send_alert(unit_title, is_critical, problems, data, action_guide.short_actions(findings, ml, 2, assessment),
                                          source=channel, assessment=assessment)
            if received_at:
                print(f"[LATENCY] {unit} {channel}: ntfy {'sent' if ok else 'FAILED'} {time.time() - received_at:.2f}s after the reading arrived", flush=True)
        except Exception as e:
            print(f"\n[ERROR] ntfy failed: {e}\n", flush=True)

    threading.Thread(target=push, name=f"ntfy-{unit}-{channel}", daemon=True).start()
    # SMS + saved alert (the saved alert carries this AC's limits and what the model found odd)
    try:
        data["_ml"] = ml
        data["_limits"] = snap.get("limits")
        data["_title"] = unit_title
        data["_critical"] = is_critical
        data["_source"] = channel
        data["_received_at"] = received_at
        process_anomaly(data)
    except Exception as e:
        print(f"\n[ERROR] process_anomaly failed: {e}\n", flush=True)


_last_health_log = {}


def _log_health(unit, ml):
    """Keep one health point per AC per minute: the model's score minus its alert line (above 0 = normal).
    This is what the Model Health page draws, and what the early-warning trend is computed from."""
    now = time.time()
    if now - _last_health_log.get(unit, 0) < 60:
        return
    _last_health_log[unit] = now
    try:
        conn = get_db_connection()
        try:
            try:
                ml_per_ac.log_health(conn, unit, ml.get("margin", ml["score"] - ml["threshold"]), ml["is_anomaly"])
            except Exception:
                conn.rollback()
                ml_per_ac.ensure_tables(conn)   # first run: create the table, then try once more
                ml_per_ac.log_health(conn, unit, ml.get("margin", ml["score"] - ml["threshold"]), ml["is_anomaly"])
            if now - _last_health_log.get("_prune", 0) > 86400:
                _last_health_log["_prune"] = now
                ml_per_ac.prune_health(conn)
        finally:
            conn.close()
    except Exception as e:
        print(f"[ML] could not log health: {e}", flush=True)


_last_prune = {"t": 0.0}


def _maybe_prune_telemetry():
    """Once a day, if a retention is set: delete live readings older than that. Only the `telemetry` table."""
    now = time.time()
    if now - _last_prune["t"] < 86400:
        return
    _last_prune["t"] = now
    days = app_settings.get_int(get_db_connection, "telemetry_retention_days", 0)
    if days <= 0:
        return
    try:
        conn = get_db_connection()
        try:
            n = telemetry_store.prune(conn, days)
        finally:
            conn.close()
        print(f"[TELEMETRY] retention {days} days: {n} old live readings removed", flush=True)
    except Exception as e:
        print(f"[TELEMETRY] retention cleanup failed: {e}", flush=True)


_SETUP_DEFAULT = {"ac_status": "Not Set", "ac_thermostat": "Not Set", "data_gathering_mode": "telemetry", "data_gathering_unit": "AC2"}
_setup_cache = {"t": 0.0, "v": None, "busy": False}


def _ac_setup_hot():
    """The AC setup WITHOUT making an alert wait for the database: once read, the last known copy is returned at once
    and refreshed in the background (every 5 s). Only the very first reading after a restart waits for the database."""
    now = time.time()
    v = _setup_cache["v"]
    if v is not None:
        if now - _setup_cache["t"] >= 5 and not _setup_cache["busy"]:
            _setup_cache["busy"] = True

            def refresh():
                try:
                    _setup_cache.update(v=get_ac_setup(), t=time.time())
                except Exception as e:
                    print(f"[SETUP] could not refresh the AC setup, using the last known one: {e}", flush=True)
                finally:
                    _setup_cache["busy"] = False
            threading.Thread(target=refresh, daemon=True).start()
        return v
    try:
        _setup_cache.update(v=get_ac_setup(), t=now)
        return _setup_cache["v"]
    except Exception as e:
        print(f"[SETUP] could not read the AC setup, using defaults: {e}", flush=True)
        return dict(_SETUP_DEFAULT)


def _units_hot():
    """Unit labels for messages, from memory only (never waits for the database)."""
    with ac_manager._cache_lock:
        units, age = ac_manager._cache["units"], time.time() - ac_manager._cache["t"]
    if units is None or age > 30:
        threading.Thread(target=lambda: ac_manager.cached_units(get_db_connection, ttl=0), daemon=True).start()
    return units or []


def _store_reading(payload_dict, ac_values, unit):
    """Write the reading to the database. Runs AFTER the alerts were decided and sent: slow or failing storage can
    no longer delay or block an alert."""
    try:
        ac_status = ac_values.get("ac_status", "Not Set")
        ac_thermostat = ac_values.get("ac_thermostat", "Not Set")
        conn = get_db_connection()
        cur = conn.cursor()
        if ac_values.get("data_gathering_mode", "telemetry") == "data_gathered":
            # TRAINING DATA: every reading is kept in full detail (never thinned)
            cur.execute(
                """
                INSERT INTO data_gathered (
                    timestamp, ac_unit, dust_sensor, dht_temp, dht_humidity, vibration,
                    ds18b20_temp1, ds18b20_temp2, pzem_voltage, pzem_current, pzem_power, pzem_energy,
                    pzem_frequency, pzem_power_factor, ac_status, ac_thermostat
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (datetime.now(LOCAL_TZ), unit, payload_dict.get("dust_sensor"), payload_dict.get("dht_temp"), payload_dict.get("dht_humidity"), payload_dict.get("vibration"), payload_dict.get("ds18b20_temp1"), payload_dict.get("ds18b20_temp2"), payload_dict.get("pzem_voltage"), payload_dict.get("pzem_current"), payload_dict.get("pzem_power"), payload_dict.get("pzem_energy"), payload_dict.get("pzem_frequency"), payload_dict.get("pzem_power_factor"), ac_status, ac_thermostat),
            )
            row_id = cur.fetchone()['id']
            print(f"\n[BACKGROUND] Stored in data_gathered with ID {row_id}", flush=True)
        else:
            # LIVE TELEMETRY (charts / dashboard): a reading that is practically the same as the last stored one is
            # skipped (a heartbeat row is still stored). The model and the safety limits already saw EVERY reading.
            now_local = datetime.now(LOCAL_TZ)
            live = dict(payload_dict, ac_status=ac_status, ac_thermostat=ac_thermostat)
            if deduper.should_store(
                unit, live, time.time(), now_local.strftime("%Y-%m-%d"),
                app_settings.get_int(get_db_connection, "telemetry_heartbeat_s", 30),
                app_settings.get_bool(get_db_connection, "telemetry_dedupe_enabled", True),
            ):
                cur.execute(
                    """
                    INSERT INTO telemetry (
                        timestamp, dust_sensor, dht_temp, dht_humidity, vibration,
                        ds18b20_temp1, ds18b20_temp2, pzem_voltage, pzem_current, pzem_power, pzem_energy,
                        pzem_frequency, pzem_power_factor, ac_status, ac_thermostat
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """,
                    (now_local.strftime("%Y-%m-%d %H:%M:%S"), payload_dict.get("dust_sensor"), payload_dict.get("dht_temp"), payload_dict.get("dht_humidity"), payload_dict.get("vibration"), payload_dict.get("ds18b20_temp1"), payload_dict.get("ds18b20_temp2"), payload_dict.get("pzem_voltage"), payload_dict.get("pzem_current"), payload_dict.get("pzem_power"), payload_dict.get("pzem_energy"), payload_dict.get("pzem_frequency"), payload_dict.get("pzem_power_factor"), ac_status, ac_thermostat),
                )
                row_id = cur.fetchone()['id']
                print(f"\n[BACKGROUND] Stored in telemetry with ID {row_id}", flush=True)
            else:
                print(f"[BACKGROUND] {unit}: unchanged live reading not stored (still scored)", flush=True)
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"\n[ERROR] Failed to save telemetry (alerts were not affected): {db_err}", flush=True)


def process_telemetry_background(payload_dict: dict):
    received_at = payload_dict.pop("_received_at", None) or time.time()
    ac_values = _ac_setup_hot()
    unit = payload_dict.get("ac_unit") or ac_values.get("data_gathering_unit", "AC2")

    # ---- 1) ALERTS FIRST, from memory only. Nothing here waits for a database write (see _store_reading below). ----
    _decide_and_send_alerts(payload_dict, ac_values, unit, received_at)

    # ---- 2) then store the reading, and the housekeeping that does need the database ----
    _store_reading(payload_dict, ac_values, unit)
    _maybe_prune_telemetry()


def _decide_and_send_alerts(payload_dict, ac_values, unit, received_at):
    """Three independent channels (ml / safety / sensor), each with its own timers. An INSTANT alert is dispatched
    here, on the reading that causes it, in its own thread."""
    from services.rule_engine import analyze_with_rules
    unit_limits = limits_store.get(get_db_connection, unit)

    model_data = dict(payload_dict)
    model_data.pop("ac_status", None)
    model_data.pop("ac_thermostat", None)

    # ML: the trained model of THIS AC scores the reading. No model (never trained, data deleted, unreadable) -> None,
    # and the ML channel stays silent: thresholds are never passed off as ML. A failure here cannot block the other channels.
    ml = None
    try:
        ml = detector.check(unit, model_data)
        if ml:
            print(f"[ML] {unit} {'ANOMALY' if ml['is_anomaly'] else 'normal'} score={ml['score']:.3f} threshold={ml['threshold']:.3f} odd={ml['top_features']}", flush=True)
            _remember_ml(unit, ml)
    except Exception as ml_err:
        ml = None
        print(f"\n[ERROR] ML scoring failed for {unit} (safety and sensor alerts continue): {ml_err}\n", flush=True)

    try:
        safety_on = app_settings.get_bool(get_db_connection, "safety_alerts_enabled", True)
        alert_payload = dict(payload_dict, ac_status=ac_values.get("ac_status", "Not Set"), ac_thermostat=ac_values.get("ac_thermostat", "Not Set"))
        channels = evaluate_channels(alert_payload, unit_limits, ml, safety_on,
                                     analyze_with_rules(alert_payload, unit_limits), ntfy_notifier.build_problems, _unit_title(unit))
    except Exception as e:
        print(f"\n[ERROR] Alert evaluation failed: {e}\n", flush=True)
        return

    now = time.time()
    for channel in CHANNELS:
        try:
            c = channels[channel]
            action = scheduler.decide(unit, channel, now, c["crit"], c["soft"], c["snapshot"])
            if action:   # own thread: a slow alert (e.g. the ML one) must never hold up an instant safety alert
                print(f"[LATENCY] {unit} {channel}: {'INSTANT' if action == 'critical' else 'normal'} alert decided {now - received_at:.2f}s after the reading arrived", flush=True)
                threading.Thread(target=_send_alert, args=(unit, channel, action == "critical", received_at),
                                 name=f"alert-{unit}-{channel}", daemon=True).start()
        except Exception as e:
            print(f"\n[ERROR] {channel} alert channel failed for {unit}: {e}\n", flush=True)

    if ml:
        _log_health(unit, ml)      # one database write a minute: after the alerts, never before


def _check_device_key(x_api_key):
	"""Only your ESP32 may post readings. Set TELEMETRY_API_KEY on Render; the ESP32 sends the same text
	in the X-API-Key header. With no TELEMETRY_API_KEY set, the check is off (as before)."""
	expected = os.environ.get("TELEMETRY_API_KEY", "")
	if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
		raise HTTPException(status_code=401, detail="Invalid device key")


@app.post("/telemetry", status_code=201)
def receive_telemetry(payload: Telemetry, background_tasks: BackgroundTasks, x_api_key: Optional[str] = Header(None)):
	_check_device_key(x_api_key)
	payload_dict = payload.dict()
	payload_dict["_received_at"] = time.time()          # so the log can show how long an alert took from the reading to the phone
	background_tasks.add_task(process_telemetry_background, payload_dict)
	return {"status": "stored"}


def get_year_month_day(timestamp):
	if isinstance(timestamp, datetime):
		return timestamp.year, timestamp.month, timestamp.day
	elif isinstance(timestamp, str):
		try:
			dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
			return dt.year, dt.month, dt.day
		except Exception:
			return int(timestamp[:4]), int(timestamp[5:7]), int(timestamp[8:10])
	else:
		now = datetime.now(LOCAL_TZ)
		return now.year, now.month, now.day


@app.get("/telemetry")
def get_telemetry(start: Optional[str] = None, end: Optional[str] = None, ac_unit: Optional[str] = 'telemetry'):
	start_param = None
	end_param = None
	try:
		if start is not None:
			start_dt = datetime.fromisoformat(start)
			start_param = start_dt.strftime("%Y-%m-%d %H:%M:%S")
		if end is not None:
			end_dt = datetime.fromisoformat(end)
			end_param = end_dt.strftime("%Y-%m-%d %H:%M:%S")
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601.")

	conn = get_db_connection()
	cur = conn.cursor()
	if ac_unit == 'telemetry' or not ac_unit:
		query = "SELECT * FROM telemetry"
		clauses = []
		params = []
		if start_param is not None:
			clauses.append("timestamp >= %s")
			params.append(start_param)
		if end_param is not None:
			clauses.append("timestamp <= %s")
			params.append(end_param)
		if clauses:
			query += " WHERE " + " AND ".join(clauses)
		query += " ORDER BY id ASC"
	else:
		query = "SELECT * FROM data_gathered"
		clauses = ["ac_unit = %s"]
		params = [ac_unit]
		if start_param is not None:
			clauses.append("timestamp >= %s")
			params.append(start_param)
		if end_param is not None:
			clauses.append("timestamp <= %s")
			params.append(end_param)
		if clauses:
			query += " WHERE " + " AND ".join(clauses)
		query += " ORDER BY id ASC"
	cur.execute(query, params)
	rows = cur.fetchall()

	results = []
	if rows:
		ymd_pairs = set()
		for r in rows:
			ymd_pairs.add(get_year_month_day(r["timestamp"]))
		
		starts = {}
		for y, m, d in ymd_pairs:
			start_str = f"{y:04d}-{m:02d}-{d:02d} 00:00:00"
			if ac_unit == 'telemetry' or not ac_unit:
				cur.execute("SELECT pzem_energy FROM telemetry WHERE timestamp >= %s ORDER BY timestamp ASC LIMIT 1", (start_str,))
			else:
				cur.execute("SELECT pzem_energy FROM data_gathered WHERE timestamp >= %s AND ac_unit = %s ORDER BY timestamp ASC LIMIT 1", (start_str, ac_unit))
			start_row = cur.fetchone()
			starts[(y, m, d)] = start_row['pzem_energy'] if (start_row and start_row['pzem_energy'] is not None) else 0.0

		for r in rows:
			d_dict = dict(r)
			y, m, d = get_year_month_day(d_dict["timestamp"])
			e_start = starts.get((y, m, d), 0.0)
			if d_dict.get("pzem_energy") is not None:
				d_dict["pzem_energy"] = max(0.0, (d_dict["pzem_energy"] - e_start) / 1000.0)
			results.append(d_dict)
	conn.close()
	return results


@app.get("/telemetry/column")
def telemetry_column(column: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None, ac_unit: Optional[str] = 'telemetry'):
	if not column:
 		raise HTTPException(status_code=400, detail="Missing 'column' query parameter")
		
	allowed = {"id", "timestamp", "dht_temp", "dht_humidity", "vibration", "ds18b20_temp1", "ds18b20_temp2", "pzem_voltage", "pzem_current", "pzem_power", "pzem_energy", "pzem_frequency", "pzem_power_factor", "ac_status", "ac_thermostat"}
	if column not in allowed:
		raise HTTPException(status_code=400, detail="Invalid column requested")

	start_param = None
	end_param = None
	try:
		if start is not None:
			start_dt = datetime.fromisoformat(start)
			start_param = start_dt.strftime("%Y-%m-%d %H:%M:%S")
		if end is not None:
			end_dt = datetime.fromisoformat(end)
			end_param = end_dt.strftime("%Y-%m-%d %H:%M:%S")
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601.")

	conn = get_db_connection()
	cur = conn.cursor()
	if ac_unit == 'telemetry' or not ac_unit:
		query = f"SELECT timestamp, {column} FROM telemetry"
		clauses = []
		params = []
		if start_param is not None:
			clauses.append("timestamp >= %s")
			params.append(start_param)
		if end_param is not None:
			clauses.append("timestamp <= %s")
			params.append(end_param)
		if clauses:
			query += " WHERE " + " AND ".join(clauses)
		query += " ORDER BY id ASC"
	else:
		query = f"SELECT timestamp, {column} FROM data_gathered"
		clauses = ["ac_unit = %s"]
		params = [ac_unit]
		if start_param is not None:
			clauses.append("timestamp >= %s")
			params.append(start_param)
		if end_param is not None:
			clauses.append("timestamp <= %s")
			params.append(end_param)
		if clauses:
			query += " WHERE " + " AND ".join(clauses)
		query += " ORDER BY id ASC"
	cur.execute(query, params)
	rows = cur.fetchall()

	results = []
	if rows:
		if column == "pzem_energy":
			ymd_pairs = set()
			for r in rows:
				ymd_pairs.add(get_year_month_day(r["timestamp"]))
			
			starts = {}
			for y, m, d in ymd_pairs:
				start_str = f"{y:04d}-{m:02d}-{d:02d} 00:00:00"
				if ac_unit == 'telemetry' or not ac_unit:
					cur.execute("SELECT pzem_energy FROM telemetry WHERE timestamp >= %s ORDER BY timestamp ASC LIMIT 1", (start_str,))
				else:
					cur.execute("SELECT pzem_energy FROM data_gathered WHERE timestamp >= %s AND ac_unit = %s ORDER BY timestamp ASC LIMIT 1", (start_str, ac_unit))
				start_row = cur.fetchone()
				starts[(y, m, d)] = start_row['pzem_energy'] if (start_row and start_row['pzem_energy'] is not None) else 0.0
			
			for r in rows:
				y, m, d = get_year_month_day(r["timestamp"])
				e_start = starts.get((y, m, d), 0.0)
				val = r[column]
				if val is not None:
					val = max(0.0, (val - e_start) / 1000.0)
				results.append({"timestamp": r["timestamp"], column: val})
		else:
			for r in rows:
				results.append({"timestamp": r["timestamp"], column: r[column]})
	conn.close()
	return results


@app.get("/telemetry/latest")
def latest_telemetry(ac_unit: Optional[str] = 'telemetry'):
	conn = get_db_connection()
	cur = conn.cursor()
	if ac_unit == 'telemetry' or not ac_unit:
		cur.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1")
	else:
		cur.execute("SELECT * FROM data_gathered WHERE ac_unit = %s ORDER BY id DESC LIMIT 1", (ac_unit,))
	row = cur.fetchone()
	if not row:
		conn.close()
		raise HTTPException(status_code=404, detail="No telemetry available")
	
	data = dict(row)
	ts = data.get("timestamp")
	y, m, d = get_year_month_day(ts)
	start_str = f"{y:04d}-{m:02d}-{d:02d} 00:00:00"
	if ac_unit == 'telemetry' or not ac_unit:
		cur.execute("SELECT pzem_energy FROM telemetry WHERE timestamp >= %s ORDER BY timestamp ASC LIMIT 1", (start_str,))
	else:
		cur.execute("SELECT pzem_energy FROM data_gathered WHERE timestamp >= %s AND ac_unit = %s ORDER BY timestamp ASC LIMIT 1", (start_str, ac_unit))
	start_row = cur.fetchone()
	conn.close()
	
	e_start = start_row['pzem_energy'] if (start_row and start_row['pzem_energy'] is not None) else 0.0
	if data.get("pzem_energy") is not None:
		data["pzem_energy"] = max(0.0, (data["pzem_energy"] - e_start) / 1000.0)
	return data


@app.get("/api/historical_data")
def get_historical_data(start: Optional[str] = None, end: Optional[str] = None):
	conn = get_db_connection()
	cur = conn.cursor()
	query = "SELECT * FROM historical_data"
	clauses = []
	params = []
	if start:
		clauses.append("date >= %s")
		params.append(start)
	if end:
		clauses.append("date <= %s")
		params.append(end)
	if clauses:
		query += " WHERE " + " AND ".join(clauses)
	query += " ORDER BY date ASC"
	cur.execute(query, params)
	rows = cur.fetchall()
	conn.close()

	result = []
	for r in rows:
		d = dict(r)
		if d.get("date") and not isinstance(d["date"], str):
			d["date"] = d["date"].strftime("%Y-%m-%d")
		result.append(d)
	return result


@app.get("/data_gathered")
def get_data_gathered(ac_unit: Optional[str] = None):
	conn = get_db_connection()
	cur = conn.cursor()
	if ac_unit:
		cur.execute("SELECT * FROM data_gathered WHERE ac_unit = %s ORDER BY timestamp DESC LIMIT 2000", (ac_unit,))
	else:
		cur.execute("SELECT * FROM data_gathered ORDER BY timestamp DESC LIMIT 2000")
	rows = cur.fetchall()

	results = []
	if rows:
		ymd_ac_pairs = set()
		for r in rows:
			y, m, d = get_year_month_day(r["timestamp"])
			ymd_ac_pairs.add((y, m, d, r["ac_unit"]))

		starts = {}
		for y, m, d, ac in ymd_ac_pairs:
			start_str = f"{y:04d}-{m:02d}-{d:02d} 00:00:00"
			cur.execute("SELECT pzem_energy FROM data_gathered WHERE timestamp >= %s AND ac_unit = %s ORDER BY timestamp ASC LIMIT 1", (start_str, ac))
			start_row = cur.fetchone()
			starts[(y, m, d, ac)] = start_row['pzem_energy'] if (start_row and start_row['pzem_energy'] is not None) else 0.0

		for r in rows:
			d_dict = dict(r)
			y, m, d = get_year_month_day(d_dict["timestamp"])
			ac = d_dict["ac_unit"]
			e_start = starts.get((y, m, d, ac), 0.0)
			if d_dict.get("pzem_energy") is not None:
				d_dict["pzem_energy"] = max(0.0, (d_dict["pzem_energy"] - e_start) / 1000.0)
			results.append(d_dict)
	conn.close()
	return results


@app.post("/data_gathered", status_code=201)
def save_data_gathered(payload: Telemetry, x_api_key: Optional[str] = Header(None)):
	_check_device_key(x_api_key)
	conn = get_db_connection()
	cur = conn.cursor()
	payload_dict = payload.dict()
	ac_unit = payload_dict.get("ac_unit") or "AC2"
	cur.execute(
		"""
		INSERT INTO data_gathered (
			timestamp, ac_unit, dust_sensor, dht_temp, dht_humidity, vibration,
			ds18b20_temp1, ds18b20_temp2, pzem_voltage, pzem_current, pzem_power, pzem_energy,
			pzem_frequency, pzem_power_factor, ac_status, ac_thermostat
		) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
		""",
		(datetime.now(LOCAL_TZ), ac_unit, payload_dict.get("dust_sensor"), payload_dict.get("dht_temp"), payload_dict.get("dht_humidity"), payload_dict.get("vibration"), payload_dict.get("ds18b20_temp1"), payload_dict.get("ds18b20_temp2"), payload_dict.get("pzem_voltage"), payload_dict.get("pzem_current"), payload_dict.get("pzem_power"), payload_dict.get("pzem_energy"), payload_dict.get("pzem_frequency"), payload_dict.get("pzem_power_factor"), payload_dict.get("ac_status"), payload_dict.get("ac_thermostat"))
	)
	row_id = cur.fetchone()['id']
	conn.commit()
	conn.close()
	return {"status": "stored", "id": row_id}


@app.get("/ac_setup")
def read_ac_setup():
	return get_ac_setup()


@app.post("/ac_setup")
def set_ac_setup(payload: ACSetup):
	ac_values = get_ac_setup()
	ac_status = payload.ac_status if payload.ac_status is not None else ac_values.get("ac_status", "Not Set")
	ac_thermostat = payload.ac_thermostat if payload.ac_thermostat is not None else ac_values.get("ac_thermostat", "Not Set")
	data_gathering_mode = payload.data_gathering_mode if payload.data_gathering_mode is not None else ac_values.get("data_gathering_mode", "telemetry")
	data_gathering_unit = payload.data_gathering_unit if payload.data_gathering_unit is not None else ac_values.get("data_gathering_unit", "AC2")
	update_ac_setup(ac_status, ac_thermostat, data_gathering_mode, data_gathering_unit)
	_setup_cache["v"] = None   # the next reading must see the new mode at once
	return {"status": "updated", "ac_status": ac_status, "ac_thermostat": ac_thermostat, "data_gathering_mode": data_gathering_mode, "data_gathering_unit": data_gathering_unit}


def _delayed_shutdown() -> None:
	time.sleep(1)
	try:
		subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
	except Exception:
		os.system("shutdown -h now")


@app.post("/system/shutdown")
def shutdown_system(request: Request):
	if not get_admin_from_token(request.cookies.get("admin_session")):
		raise HTTPException(status_code=401, detail="Login required")
	threading.Thread(target=_delayed_shutdown, daemon=True).start()
	return {"status": "accepted", "message": "Shutdown command sent."}


@app.get("/api/isolation_forest/scatter")
def get_isolation_forest_scatter(ac_unit: Optional[str] = None, limit: int = 500):
	"""2D scatter for the Isolation Forest page; every AC is scored by its own model."""
	conn = get_db_connection()
	try:
		return ml_per_ac.scatter_from_db(conn, detector.get_bundles(), ac_unit or None, max(10, min(limit, 2000)))
	except Exception as e:
		return {"error": str(e), "points": [], "stats": {"total": 0, "regular": 0, "abnormal": 0, "contamination": 0.01}}
	finally:
		conn.close()


@app.get("/api/ml/overview")
def ml_overview(ac_unit: str):
	"""Everything the Model Health page shows for one AC, from real data only."""
	from datetime import timedelta
	out = {"ac_unit": ac_unit, "model": None, "profile": [], "health": [], "latest": None, "trend": None, "learning": []}
	bundle = detector.get_bundles().get(ac_unit)
	if bundle:
		out["model"] = {"n_rows": bundle["n_rows"], "trained_at": bundle.get("trained_at"), "pct": bundle.get("pct")}
		for feat in ml_per_ac.FEATURES:
			p = (bundle.get("profile") or {}).get(feat)
			if p:
				out["profile"].append({"name": action_guide.FRIENDLY.get(feat, feat), "unit": ml_per_ac.UNITS.get(feat, ""),
										"low": round(p["low"], 2), "mid": round(p["mid"], 2), "high": round(p["high"], 2)})
	conn = get_db_connection()
	try:
		try:
			out["learning"] = ml_per_ac.fetch_curve(conn, ac_unit)
		except Exception as e:
			conn.rollback()
			print(f"[ML] overview curve: {e}", flush=True)
		try:
			df = ml_per_ac.fetch_health(conn, ac_unit, days=30)
			if len(df):
				df["ts"] = ml_per_ac.pd.to_datetime(df["ts"], utc=True)
				last = df.iloc[-1]
				out["latest"] = {"t": last["ts"].isoformat(), "margin": float(last["margin"]), "flagged": bool(last["flagged"])}
				recent = df[df["ts"] >= df["ts"].max() - timedelta(days=3)]
				out["health"] = ml_per_ac.bucket_health(recent)
			out["trend"] = ml_per_ac.trend(df)
		except Exception as e:
			conn.rollback()
			print(f"[ML] overview health: {e}", flush=True)
		return out
	finally:
		conn.close()


def _ml_status_for(code):
    """What the dashboard shows for one AC: the model's own verdict on the newest reading."""
    safety_on = app_settings.get_bool(get_db_connection, "safety_alerts_enabled", True)
    out = {"ac_unit": code, "state": "select_unit", "safety_alerts_enabled": safety_on, "alerting_gap": False}
    units = ac_manager.cached_units(get_db_connection) or []
    unit = next((u for u in units if u["code"] == code), None)
    if unit is None:
        out["message"] = "Select an AC unit to see its ML detection result."
        return out
    out["label"] = unit["label"]

    bundle = detector.bundle_info(code)
    load_error = detector.load_error(code)
    if bundle is None:
        retrain = bool(unit.get("needs_retrain") or load_error)
        out["state"] = "retrain_required" if retrain else "no_model"
        out["message"] = (
            "The stored model of this AC could not be loaded. Retrain it on Manage ACs." if load_error else
            "The training data of this AC was deleted. Collect new readings and press Train now: no ML detection until then." if retrain else
            "No trained model for this AC yet: no ML detection. Collect 200+ readings and press Train now.")
        out["alerting_gap"] = not safety_on          # no model AND safety limits off: only sensor-offline alerts remain
        return out

    out.update({"trained_at": bundle["trained_at"], "n_rows": bundle["n_rows"]})
    result, checked_at, source, stale = None, None, None, False
    with _latest_ml_lock:
        live = _latest_ml.get(code)
    if live and time.time() - live["epoch"] <= 120:
        result, checked_at, source = live["ml"], live["at"], "live"
    else:
        try:   # no fresh live verdict (restart, device quiet): score the newest stored reading with the same model
            conn = get_db_connection()
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT timestamp, {', '.join(ml_per_ac.FEATURES)} FROM data_gathered WHERE ac_unit = %s ORDER BY id DESC LIMIT 1", (code,))
                row = cur.fetchone()
            finally:
                conn.close()
            if row:
                result = detector.check(code, dict(row))
                ts = row["timestamp"]
                checked_at, source = (ts.isoformat() if hasattr(ts, "isoformat") else str(ts)), "latest_stored"
        except Exception as e:
            print(f"[ML] status: could not score the newest stored reading: {e}", flush=True)
        if result is None and live:
            result, checked_at, source, stale = live["ml"], live["at"], "live", True
    if result is None:
        out["state"] = "waiting"
        out["message"] = "Model is ready. Waiting for the next reading from this AC."
        return out

    out.update({
        "state": ml_state(result),
        "score": round(float(result["score"]), 4),
        "threshold": round(float(result["threshold"]), 4),
        "margin": round(float(result.get("margin", result["score"] - result["threshold"])), 4),
        "odd_features": [{"key": k, "name": action_guide.FRIENDLY.get(k, k), "z": z} for k, z in (result.get("top_features") or [])],
        "checked_at": checked_at, "source": source, "stale": stale,
        "operating_mode": result.get("operating_mode"), "distance_ratio": result.get("distance_ratio"), "checks": result.get("checks"),
    })
    if result.get("is_anomaly"):   # the same explanation an ML alert carries: severity, odd sensors with values, risks, actions
        a = ml_explain.assess(result, None, unit["label"])
        if a:
            out["assessment"] = {k: a[k] for k in ("severity", "condition", "headline", "why", "risks", "actions")}
            out["odd_sensors"] = [{k: o[k] for k in ("name", "value_text", "usual_text", "compare", "severity", "means")} for o in a["odd"]]
    return out


@app.get("/api/ml/status")
def ml_status(request: Request, ac_unit: str = ""):
    """The trained model's verdict for one AC: normal / anomaly / no model / retraining required."""
    if not get_admin_from_token(request.cookies.get("admin_session")):
        raise HTTPException(status_code=401, detail="Login required")
    return _ml_status_for(ac_unit)


@app.get("/api/ml/status/all")
def ml_status_all(request: Request):
    """Live model verdict of every AC that sent a reading in the last 2 minutes (no database scoring: cheap enough to poll)."""
    if not get_admin_from_token(request.cookies.get("admin_session")):
        raise HTTPException(status_code=401, detail="Login required")
    units = {u["code"]: u for u in (ac_manager.cached_units(get_db_connection) or [])}
    out = []
    with _latest_ml_lock:
        live = dict(_latest_ml)
    for code, item in live.items():
        if time.time() - item["epoch"] > 120:
            continue
        ml = item["ml"]
        label = (units.get(code) or {}).get("label", code)
        row = {"ac_unit": code, "label": label, "state": ml_state(ml), "checked_at": item["at"]}
        a = ml_explain.assess(ml, None, label) if ml.get("is_anomaly") else None
        if a:
            row.update({"severity": a["severity"], "condition": a["condition"], "headline": a["headline"]})
        out.append(row)
    return out


@app.get("/api/version")
def version():
	"""Which version of this code is running (also shown in the sidebar)."""
	return {"version": app_version()}


@app.get("/api/ntfy/test")
def ntfy_test(request: Request):
	"""Send a test notification to your phone and say exactly what happened (also shows why it failed)."""
	if not get_admin_from_token(request.cookies.get("admin_session")):
		raise HTTPException(status_code=401, detail="Login required")
	sent = ntfy_notifier.send_test()
	if not sent and os.environ.get("NTFY_TOPIC"):
		ntfy_notifier.diagnose_network()       # why it failed: DNS, no route, or only one address family (stored in the status)
	return {"sent": sent, "topic_configured": bool(os.environ.get("NTFY_TOPIC")), **ntfy_notifier.status()}


@app.get("/api/ntfy/status")
def ntfy_status(request: Request):
	"""Is ntfy set up, and did the last notification arrive? If not, the reason."""
	if not get_admin_from_token(request.cookies.get("admin_session")):
		raise HTTPException(status_code=401, detail="Login required")
	return ntfy_notifier.status()
