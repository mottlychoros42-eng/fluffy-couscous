"""
Tests for the v3.0 alerting pipeline. Run from the project root:

    python tests/test_ml_alerting.py          (or: python -m pytest tests -q)

The ML part is REAL: per-AC models are trained with ml_per_ac.train_unit() on generated normal data and every reading is
scored by ml_per_ac.score_reading() / LiveDetector.check(). Nothing sets an "is_anomaly" flag by hand.
The web framework and the database are stubbed, so this needs only pandas, numpy and scikit-learn.
"""
import collections
import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---- stubs for packages / modules that are not needed to test the logic -------------------------------
fastapi = types.ModuleType("fastapi")


class _Router:
    def __init__(self, *a, **k): pass
    def _d(self, *a, **k): return lambda f: f
    get = post = put = delete = _d


class _HTTPException(Exception):
    def __init__(self, status_code=500, detail=""): self.status_code, self.detail = status_code, detail


for _n, _v in dict(APIRouter=_Router, HTTPException=_HTTPException, Request=object, Response=object, BackgroundTasks=object).items():
    setattr(fastapi, _n, _v)
sys.modules.setdefault("fastapi", fastapi)

import numpy as np
import pandas as pd

from services import ml_per_ac, rule_engine, limits, action_guide, ntfy_notifier
from services import alert_channels as ch
from services import settings as app_settings
from services import telemetry_store

CAPTURED = {"alerts": [], "sms": [], "prompts": [], "events": [], "updates": []}


def _ask(prompt):
    CAPTURED["prompts"].append(prompt)
    CAPTURED["events"].append("ask")
    raise RuntimeError("no LLM in tests")


def _send_sms(text):
    CAPTURED["sms"].append(text)
    CAPTURED["events"].append("sms")


sys.modules["services.ai_predictor"] = types.SimpleNamespace(ask=_ask)
sys.modules["services.traccar_sms"] = types.SimpleNamespace(send_sms=_send_sms)


class _AlertIn:
    def __init__(self, summary, diagnoses=None): self.summary, self.diagnoses = summary, diagnoses


sys.modules["services.alerts_api"] = types.SimpleNamespace(
    AlertIn=_AlertIn,
    create_alert=lambda a: (CAPTURED["alerts"].append(a), CAPTURED["events"].append("save"), types.SimpleNamespace(id=len(CAPTURED["alerts"])))[2],
    update_alert_diagnoses=lambda i, d: (CAPTURED["updates"].append((i, d)), CAPTURED["events"].append("update")))
from services import anomaly_reciever  # noqa: E402


# ---- real models trained on generated "normal" data ------------------------------------------------
def make_normal(n, seed, current, power, vib, outlet, inlet, temp, hum=85.0, idle_share=0.15):
    """Normal behaviour of one AC: mostly running, sometimes the compressor is off between cycles."""
    rng = np.random.default_rng(seed)
    run = rng.random(n) > idle_share
    df = pd.DataFrame({
        "dht_temp": np.where(run, rng.normal(temp, 0.4, n), rng.normal(27, 0.3, n)),
        "dht_humidity": np.where(run, rng.normal(hum, 2.0, n), rng.normal(60, 2.0, n)),
        "vibration": np.where(run, rng.normal(vib, 3.0, n), rng.normal(2, 0.5, n)),
        "ds18b20_temp1": np.where(run, rng.normal(outlet, 1.0, n), rng.normal(30, 0.5, n)),
        "ds18b20_temp2": np.where(run, rng.normal(inlet, 0.5, n), rng.normal(26, 0.5, n)),
        "pzem_voltage": rng.normal(233, 1.0, n),
        "pzem_current": np.where(run, rng.normal(current, 0.15, n), rng.normal(0.1, 0.02, n)),
        "pzem_power": np.where(run, rng.normal(power, 20, n), rng.normal(5, 1, n)),
        "pzem_frequency": rng.normal(60.0, 0.05, n),
        "pzem_power_factor": np.where(run, rng.normal(0.94, 0.01, n), rng.normal(0.4, 0.02, n)),
    })
    return df


# AC-A is a 1.5 HP unit like the defaults, AC-B is a smaller unit with its own normal
DATA = {
    "AC-A": make_normal(3000, 1, current=8.0, power=1750, vib=55, outlet=55, inlet=11, temp=16),
    "AC-B": make_normal(3000, 2, current=6.0, power=1300, vib=40, outlet=52, inlet=10, temp=17),
}
BUNDLES = {u: ml_per_ac.train_unit(df) for u, df in DATA.items()}
assert all(b is not None for b in BUNDLES.values()), "training must work on the generated data"

NORMAL_A = dict(dht_temp=16.1, dht_humidity=85.0, vibration=55.0, ds18b20_temp1=55.2, ds18b20_temp2=11.1,
                pzem_voltage=233.0, pzem_current=8.0, pzem_power=1750.0, pzem_frequency=60.0, pzem_power_factor=0.94)
# every one of these 8 sensors is INSIDE the default safety limits, yet the combination is nothing like AC-A's normal
ODD_BUT_IN_LIMITS = dict(dht_temp=24.0, dht_humidity=85.0, vibration=85.0, ds18b20_temp1=60.0, ds18b20_temp2=16.0,
                         pzem_voltage=233.0, pzem_current=8.7, pzem_power=1660.0, pzem_frequency=60.0, pzem_power_factor=0.94)
NORMAL_B = dict(dht_temp=17.0, dht_humidity=85.0, vibration=40.0, ds18b20_temp1=52.0, ds18b20_temp2=10.0,
                pzem_voltage=233.0, pzem_current=6.0, pzem_power=1300.0, pzem_frequency=60.0, pzem_power_factor=0.94)


def detector(bundles=None):
    d = ml_per_ac.LiveDetector(lambda: (_ for _ in ()).throw(RuntimeError("database not used in tests")))
    d._bundles = dict(BUNDLES if bundles is None else bundles)
    d._loaded_at = time.time()
    return d


def judge(reading, unit="AC-A", safety_on=True, det=None):
    det = det or detector()
    L = limits.merged(None)
    ml = det.check(unit, reading)
    rules = rule_engine.analyze_with_rules(reading, L)
    return ml, ch.evaluate_channels(dict(reading), L, ml, safety_on, rules, ntfy_notifier.build_problems)


# ---- 1. the model really decides -----------------------------------------------------------------
def test_normal_reading_is_normal():
    ml, c = judge(NORMAL_A)
    assert ml is not None and ml["is_anomaly"] is False, ml
    assert not c["ml"]["soft"] and not c["safety"]["soft"] and not c["safety"]["crit"]


def test_ml_flags_abnormal_reading_that_is_inside_every_safety_limit():
    L = limits.merged(None)
    assert rule_engine.analyze_with_rules(ODD_BUT_IN_LIMITS, L)["findings"] == [], "must be inside the limits for this test"
    assert not ch.limit_breaches(ODD_BUT_IN_LIMITS, ch.limit_sets(L))
    ml, c = judge(ODD_BUT_IN_LIMITS, safety_on=True)
    assert ml["is_anomaly"] is True and ml["margin"] < 0, ml
    assert c["ml"]["soft"] and c["ml"]["snapshot"]["source"] == "ml"
    assert not c["safety"]["soft"] and not c["safety"]["crit"], "no limit is broken, so no safety alert"
    # and the same reading with the safety switch OFF: the ML alert is unchanged
    _, c_off = judge(ODD_BUT_IN_LIMITS, safety_on=False)
    assert c_off["ml"]["soft"]


def test_each_ac_is_judged_by_its_own_model():
    det = detector()
    on_b = det.check("AC-B", NORMAL_B)
    on_a = det.check("AC-A", NORMAL_B)          # AC-B's normal behaviour shown to AC-A's model
    assert on_b["is_anomaly"] is False
    assert on_a["is_anomaly"] is True
    assert det.check("AC-A", NORMAL_A)["is_anomaly"] is False
    assert det.check("AC-B", NORMAL_A)["is_anomaly"] is True


def test_result_carries_what_the_dashboard_shows():
    ml = detector().check("AC-A", ODD_BUT_IN_LIMITS)
    for k in ("score", "threshold", "margin", "is_anomaly", "top_features", "trained_at", "n_rows"):
        assert k in ml, k
    assert ml["top_features"], "the model should name the odd sensors"


# ---- 2. no model: no ML alert, never a threshold dressed up as ML --------------------------------------
def test_no_model_means_no_ml_alert_and_no_fallback_to_rules():
    det = detector({})                                  # AC without a model
    breaking = dict(NORMAL_A, pzem_current=12.0, pzem_power=2500.0)
    ml, c = judge(breaking, det=det)
    assert ml is None and ch.ml_state(ml) == "no_model"
    assert not c["ml"]["soft"] and c["ml"]["snapshot"] is None
    assert c["safety"]["crit"], "the safety channel still protects the AC on its own"


def test_drop_removes_the_model_immediately_even_if_the_database_is_down():
    det = detector()
    assert det.check("AC-A", NORMAL_A) is not None
    det.drop("AC-A")
    det._loaded_at = time.time()      # pretend a refresh just happened and failed: the old copy must not come back
    assert det.check("AC-A", NORMAL_A) is None
    assert det.check("AC-B", NORMAL_B) is not None, "other ACs keep their model"


# ---- 3. safety switch ----------------------------------------------------------------------------------
def test_safety_switch_off_stops_limit_alerts_only():
    both = dict(ODD_BUT_IN_LIMITS, pzem_current=9.6, pzem_power=1950.0, vibration=95.0)     # limits broken AND off-pattern
    _, on = judge(both, safety_on=True)
    _, off = judge(both, safety_on=False)
    assert on["safety"]["crit"] and on["ml"]["soft"]
    assert not off["safety"]["crit"] and not off["safety"]["soft"] and off["safety"]["snapshot"] is None
    assert off["ml"]["soft"], "ML alerts do not depend on the switch"
    assert off["ml"]["snapshot"]["findings"] == [], "an ML alert carries no threshold findings"


def test_safety_channel_ignores_the_model():
    breaking = dict(NORMAL_A, pzem_current=9.6)
    _, c = judge(breaking)                               # the model may say anything: safety only reads the limits
    assert c["safety"]["crit"] and c["safety"]["snapshot"]["ml"] is None


def test_sensor_offline_has_its_own_channel_unaffected_by_both_switches():
    reading = dict(NORMAL_A, dht_temp=None, dht_humidity=None)
    for safety_on in (True, False):
        _, c = judge(reading, safety_on=safety_on)
        assert c["sensor"]["soft"], safety_on
        assert all(f["sensor_key"] in ch.OFFLINE_RULE_IDS for f in c["sensor"]["snapshot"]["findings"])
        assert not any(f["sensor_key"] in ch.OFFLINE_RULE_IDS for f in (c["safety"]["snapshot"] or {}).get("findings", []))


# ---- 4. the channels never block each other (timers) ---------------------------------------------------
def simulate(minutes, safety_problem, ml_problem, safety_on=True, step=10):
    sch = ch.AlertScheduler()
    fired = {"ml": [], "safety": []}
    for i in range(int(minutes * 60 / step) + 1):
        t = i * step
        crit = bool(safety_problem and safety_on)
        a = sch.decide("AC-A", "safety", t, crit, False, {"payload": {}} if crit else None)
        if a: fired["safety"].append((t, a))
        b = sch.decide("AC-A", "ml", t, False, bool(ml_problem), {"payload": {}} if ml_problem else None)
        if b: fired["ml"].append((t, b))
    return fired


def test_urgent_safety_alert_does_not_delay_or_block_the_ml_alert():
    f = simulate(minutes=8, safety_problem=True, ml_problem=True)
    assert f["safety"][0] == (0, "critical"), f["safety"]
    assert [t for t, _ in f["safety"]][:3] == [0, 120, 240], "urgent repeats every 2 minutes"
    assert f["ml"] and f["ml"][0][0] == 180, f"ML alert comes after its own 3 minutes, not held back by safety: {f['ml']}"
    assert f["ml"][0][1] == "soft"


def test_ml_alert_with_safety_switch_off():
    f = simulate(minutes=8, safety_problem=True, ml_problem=True, safety_on=False)
    assert f["safety"] == []
    assert [t for t, _ in f["ml"]][:2] == [180, 360]


def test_a_short_ml_blip_does_not_alert():
    sch = ch.AlertScheduler()
    got = []
    for i in range(0, 19):
        t = i * 10
        got.append(sch.decide("AC-A", "ml", t, False, 30 <= t < 60, {"payload": {}} if 30 <= t < 60 else None))
    assert not any(got)


# ---- 5. humidity -----------------------------------------------------------------------------------------
def test_high_humidity_never_alerts_low_humidity_does():
    L = limits.merged(None)
    for hum in (81, 90, 99.9, 100):
        r = rule_engine.analyze_with_rules(dict(NORMAL_A, dht_humidity=hum), L)
        assert not [f for f in r["findings"] if "umid" in f["sensor"] or "umid" in f["issue"]], hum
        _, c = judge(dict(NORMAL_A, dht_humidity=hum), safety_on=True)
        assert not c["safety"]["soft"] and not c["safety"]["crit"], hum
    low = rule_engine.analyze_with_rules(dict(NORMAL_A, dht_humidity=12.0), L)
    ids = [f["sensor_key"] for f in low["findings"]]
    assert ids == ["low_humidity"], ids
    assert "dht_humidity" not in [r["id"] for r in rule_engine.build_rules(L)], "the old high-humidity rule is gone"


def test_humidity_limit_is_low_side_only():
    assert limits.DEFAULT_LIMITS["humidity"]["min"] > 0
    meta = next(m for m in limits.LIMIT_META if m["key"] == "humidity")
    assert meta["use"] == "min"
    clean, err = limits.validate({m["key"]: dict(limits.DEFAULT_LIMITS[m["key"]]) for m in limits.LIMIT_META})
    assert err is None


def test_llm_prompt_does_not_ask_for_high_humidity():
    p = anomaly_reciever.generate_combined_prompt({"Return Air Humidity (%)": 95})
    assert "<= 80%" not in p and "do NOT report high humidity" in p


def test_ml_hint_for_humidity_is_not_about_the_drain():
    assert "drain" not in action_guide.FEATURE_HINTS["dht_humidity"].lower()
    assert "low_humidity" in action_guide.GUIDE and "dht_humidity" not in action_guide.GUIDE


# ---- 6. alerts are distinguishable ---------------------------------------------------------------------
def _run_process_anomaly(source, reading, ml):
    for k in CAPTURED: CAPTURED[k].clear()
    data = dict(reading, ac_status="Cool", ac_thermostat="22")
    data.update({"_ml": ml, "_limits": limits.merged(None), "_title": "AC1", "_critical": False, "_source": source})
    anomaly_reciever.process_anomaly(data)
    saved = json.loads(CAPTURED["alerts"][0].diagnoses)
    return saved, CAPTURED["alerts"][0].summary, CAPTURED["sms"][0]


def test_ml_alert_is_labelled_and_has_no_threshold_content():
    ml = detector().check("AC-A", ODD_BUT_IN_LIMITS)
    saved, summary, sms = _run_process_anomaly("ml", dict(ODD_BUT_IN_LIMITS, pzem_current=9.6), ml)   # even with a limit broken
    assert saved["source"] == "ml" and saved["source_label"] == "ML"
    fs = saved["rule_diagnoses"]["findings"]
    assert fs and all(f["origin"] == "ml" for f in fs), "an ML alert lists what the MODEL found, never threshold findings"
    assert not any(f["sensor_key"].startswith(("individual_", "high_", "low_", "excessive_")) for f in fs)
    assert saved["ml"]["is_anomaly"] is True
    assert "[ML]" in summary and "[ML]" in sms and "ML [" in summary and "usual" in summary
    assert "machine-learning model of this AC" in CAPTURED["prompts"][0]


def test_safety_alert_is_labelled_and_never_carries_model_output():
    ml = detector().check("AC-A", ODD_BUT_IN_LIMITS)
    saved, summary, sms = _run_process_anomaly("safety", dict(NORMAL_A, pzem_current=9.6), ml)
    assert saved["source"] == "safety" and saved["ml"] is None
    assert [f["sensor_key"] for f in saved["rule_diagnoses"]["findings"]] == ["individual_current_high"]
    assert "[SAFETY]" in sms and "ML model" not in summary


def test_sensor_alert_only_lists_offline_findings():
    saved, summary, sms = _run_process_anomaly("sensor", dict(NORMAL_A, dht_temp=None, dht_humidity=None, pzem_current=9.6), None)
    assert saved["source"] == "sensor"
    assert [f["sensor_key"] for f in saved["rule_diagnoses"]["findings"]] == ["dht_offline"]
    assert "[SENSOR]" in sms


def test_ntfy_title_says_the_source():
    t_ml, _ = ntfy_notifier.build_message("AC1", False, ["ML model: x"], source="ml")
    t_sf, _ = ntfy_notifier.build_message("AC1", True, ["Current is too high"], source="safety")
    assert t_ml == "Check (ML): AC1" and t_sf == "STOP (Safety): AC1"


# ---- 7. Delete Data touches only one AC's training state ----------------------------------------------
class FakeCursor:
    def __init__(self, log, rowcount_for): self.log, self.rc, self.rowcount = log, rowcount_for, 0
    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))
        self.rowcount = self.rc.get(sql.split()[2] if sql.startswith("DELETE") else "", 1)
    def fetchall(self): return []
    def fetchone(self): return {}


class FakeConn:
    def __init__(self, rc=None): self.log, self.rc, self.commits, self.rollbacks = [], rc or {}, 0, 0
    def cursor(self): return FakeCursor(self.log, self.rc)
    def commit(self): self.commits += 1
    def rollback(self): self.rollbacks += 1


def test_delete_training_data_is_scoped_to_one_ac_and_never_touches_telemetry():
    from services import ac_manager
    ac_manager._ready = True
    conn = FakeConn({"data_gathered": 1234, "ac_models": 1})
    res = ac_manager.delete_training_data(conn, "AC2")
    assert res == {"readings_deleted": 1234, "model_removed": True}
    touched = []
    for sql, params in conn.log:
        assert params == ("AC2",), (sql, params)                  # every statement filtered by that one AC
        assert "telemetry" not in sql.lower() and "alerts" not in sql.lower() and "ac_limits" not in sql.lower()
        if sql.startswith("DELETE FROM "):
            touched.append(sql.split()[2])
    assert tuple(touched) == ac_manager.TRAINING_DATA_TABLES
    assert any("UPDATE ac_units SET data_deleted_at" in s for s, _ in conn.log) and conn.commits == 1
    assert "ac_units WHERE" not in " ".join(s for s, _ in conn.log if s.startswith("DELETE")), "the AC itself stays"


def test_failed_delete_rolls_back_everything():
    from services import ac_manager
    ac_manager._ready = True
    conn = FakeConn()
    orig = conn.cursor
    calls = {"n": 0}

    class Boom(FakeCursor):
        def execute(self, sql, params=None):
            calls["n"] += 1
            if calls["n"] == 3: raise RuntimeError("db error")
            super().execute(sql, params)
    conn.cursor = lambda: Boom(conn.log, {})
    try:
        ac_manager.delete_training_data(conn, "AC2"); assert False
    except RuntimeError:
        pass
    assert conn.rollbacks == 1 and conn.commits == 0


def test_training_status_after_deletion():
    from services import ac_manager
    ts = ac_manager.training_status
    assert ts(True, 5000, None) == "trained"
    assert ts(False, 0, "2026-11-06T10:00:00") == "retrain_required"       # deleted: retraining required...
    assert ts(False, 9000, "2026-11-06T10:00:00") == "retrain_required"    # ...even if new data has arrived since
    assert ts(False, 500, None) == "ready_to_train" and ts(False, 50, None) == "needs_data"


# ---- 8. telemetry thinning -----------------------------------------------------------------------------
def test_deduper_skips_unchanged_but_keeps_changes_heartbeat_and_new_day():
    d = telemetry_store.Deduper()
    r = dict(NORMAL_A, ac_status="Cool", ac_thermostat="22")
    assert d.should_store("AC1", r, 0, "2026-11-06", 30)                                   # first one
    assert not d.should_store("AC1", dict(r, pzem_voltage=233.4, dht_temp=16.2), 10, "2026-11-06", 30)   # wobble: skipped
    assert d.should_store("AC1", dict(r, pzem_current=8.5), 20, "2026-11-06", 30)          # real change: stored
    assert not d.should_store("AC1", dict(r, pzem_current=8.52), 30, "2026-11-06", 30)
    assert d.should_store("AC1", dict(r, pzem_current=8.52), 55, "2026-11-06", 30)         # heartbeat
    assert d.should_store("AC1", dict(r, pzem_current=8.52), 60, "2026-11-07", 30)         # new day (daily energy starts here)
    assert d.should_store("AC1", dict(r, pzem_current=8.52, dht_temp=None), 65, "2026-11-07", 30)   # sensor went offline
    assert d.should_store("AC2", r, 66, "2026-11-07", 30), "each device has its own history"
    off = telemetry_store.Deduper()
    assert off.should_store("AC1", r, 0, "d", 30, enabled=False) and off.should_store("AC1", r, 1, "d", 30, enabled=False)


def test_prune_only_touches_the_telemetry_table():
    conn = FakeConn({"telemetry": 7})
    assert telemetry_store.prune(conn, 30) == 7
    sql = " ".join(s for s, _ in conn.log)
    assert sql.startswith("DELETE FROM telemetry WHERE") and "data_gathered" not in sql and "ac_models" not in sql
    assert telemetry_store.prune(FakeConn(), 0) == 0


# ---- 9. settings ---------------------------------------------------------------------------------------
def test_settings_fail_safe_and_validation():
    app_settings.invalidate()
    def bad(): raise RuntimeError("down")
    assert app_settings.get_bool(bad, "safety_alerts_enabled", True) is True, "a database problem must never switch safety alerts off"
    assert app_settings.normalise("safety_alerts_enabled", False) == ("false", None)
    assert app_settings.normalise("safety_alerts_enabled", "maybe")[1]
    assert app_settings.normalise("telemetry_retention_days", -1)[1]
    assert app_settings.normalise("nope", 1)[1]


# =====================================================================================================
# v3.1: richer ML alerts, condition, ntfy, instant alerts
# =====================================================================================================
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fault_data  # noqa: E402
from services import ml_explain  # noqa: E402

SIM_BUNDLE = ml_per_ac.train_unit(fault_data.normal_rows(4000))
SIM_DET = detector({"SIM": SIM_BUNDLE})
SCEN = fault_data.scenarios()


def _score_all(rows):
    return [SIM_DET.check("SIM", r) for r in rows]


# ---- 1/2. real abnormal datasets: detection, severity and condition ---------------------------------------
def test_every_fault_dataset_is_detected_by_real_inference():
    for name, rows in SCEN.items():
        res = _score_all(rows)
        rate = sum(r["is_anomaly"] for r in res) / len(res)
        assert rate >= 0.95, (name, rate)


def test_normal_data_rarely_flags_and_never_produces_an_alert_after_confirmation():
    fresh = fault_data.normal_rows(2000, seed=99, start=1e6).to_dict("records")
    res = _score_all(fresh)
    assert sum(r["is_anomaly"] for r in res) / len(res) < 0.06
    sch = ch.AlertScheduler()
    fired = []
    for i, r in enumerate(res):
        fired.append(sch.decide("SIM", "ml", i * 10, False, r["is_anomaly"], {"payload": {}} if r["is_anomaly"] else None))
    assert not any(fired), "isolated false alarms are absorbed by the 3-minute confirmation"


def test_severity_follows_how_abnormal_the_readings_really_are():
    sev = {}
    for name, rows in SCEN.items():
        a = [ml_explain.assess(r, x, "SIM") for r, x in zip(_score_all(rows), rows)]
        order = ["Low", "Medium", "High", "Critical"]
        sev[name] = order[sorted(order.index(x["severity"]) for x in a if x)[len(a) // 2]]   # median severity
    assert sev["sim_fault (over every limit)"] == "Critical"
    assert sev["compressor stuck off while cooling"] == "Critical"
    assert sev["refrigerant leak"] in ("High", "Critical")
    assert sev["overvoltage 250 V"] in ("High", "Critical") and sev["undervoltage 205 V"] in ("High", "Critical")
    assert sev["sim_drift_peak (inside limits)"] in ("Medium", "High")
    assert sev["sim_drift_half (inside limits)"] in ("Low", "Medium")
    assert sev["bearing wear (vibration only)"] in ("Medium", "High")


def test_condition_is_never_good_for_an_abnormal_reading():
    for name, rows in SCEN.items():
        for ml, row in zip(_score_all(rows), rows):
            a = ml_explain.assess(ml, row, "SIM")
            assert a and a["condition"] in ("Need Repair", "Check Power Supply", "Need Maintenance"), (name, a and a["condition"])
    fault = ml_explain.assess(SIM_DET.check("SIM", SCEN["sim_fault (over every limit)"][0]), SCEN["sim_fault (over every limit)"][0], "SIM")
    assert fault["condition"] == "Need Repair"
    volt = [ml_explain.assess(m, r, "SIM") for m, r in zip(_score_all(SCEN["overvoltage 250 V"]), SCEN["overvoltage 250 V"])]
    assert all(a["condition"] in ("Check Power Supply", "Need Repair") for a in volt)


def test_ml_alert_has_the_detail_of_a_rule_alert():
    row = SCEN["sim_fault (over every limit)"][3]
    ml = SIM_DET.check("SIM", row)
    a = ml_explain.assess(ml, row, "AC1")
    keys = {o["key"] for o in a["odd"]}
    assert {"pzem_current", "pzem_voltage", "pzem_power", "vibration"} <= keys, keys
    cur = next(o for o in a["odd"] if o["key"] == "pzem_current")
    assert cur["value"] == row["pzem_current"] and cur["unit"] == "A" and cur["direction"] == "above"       # the real value
    assert cur["usual_low"] < 8.15 < cur["usual_high"] and "–" in cur["usual_text"]                          # this AC's usual range
    assert "Overload" in cur["risk"] or "overload" in cur["risk"].lower()
    assert a["severity"] == "Critical" and a["risks"] and a["actions"] and a["steps"]
    assert str(row["pzem_current"]) in a["why"] or f"{row['pzem_current']:.2f}" in a["why"]                  # explains with the numbers
    assert "own history" in a["why"] and "not with a fixed safety limit" in a["why"]
    for f in a["findings"]:                                                                                   # same fields the page reads
        for k in ("sensor", "sensor_key", "measured_value", "unit", "normal_range", "issue", "severity", "root_cause", "recommended_action"):
            assert k in f, k
        assert f["origin"] == "ml"
    for st in a["steps"]:
        for k in ("title", "severity", "urgency", "short", "why", "do_now", "check", "call_technician_if"):
            assert k in st, k


def test_ml_alert_with_no_single_odd_sensor_still_explains_itself():
    ml = {"is_anomaly": True, "margin": -0.2, "distance_ratio": 2.0, "checks": ["forest"], "operating_mode": "running", "deviations": [], "n_rows": 500}
    a = ml_explain.assess(ml, {}, "AC1")
    assert a["odd"] == [] and a["findings"][0]["sensor_key"] == "ml_pattern" and a["severity"] == "Low" and a["condition"] == "Need Maintenance"
    assert "together" in a["why"] and a["steps"] and a["actions"]


def test_idle_compressor_is_compared_with_idle_not_with_running():
    idle = dict(NORMAL_A, pzem_current=0.1, pzem_power=5.0, vibration=2.0, ds18b20_temp1=30.0, ds18b20_temp2=26.0, dht_temp=27.0, dht_humidity=60.0, pzem_power_factor=0.4)
    ml = detector().check("AC-A", idle)
    assert ml["operating_mode"] == "idle"
    assert ml["is_anomaly"] is False and not ml["deviations"], "a normal idle reading must not look odd next to running values"


def test_ml_alert_sms_and_ntfy_carry_values_severity_risk_and_actions():
    row = SCEN["sim_fault (over every limit)"][3]
    ml = SIM_DET.check("SIM", row)
    a = ml_explain.assess(ml, row, "AC1")
    sms = action_guide.sms_text("AC1", False, a["findings"], ml, "ml", a)
    assert "[ML]" in sms and "ML [Critical]" in sms and "usual" in sms and "Do now:" in sms
    assert "supply voltage too high" in sms and "voltage 250" in sms, "the SMS leads with the likely cause and the sensor that defines it, with its real value"
    assert all(ord(c) < 128 for c in sms) and len(sms) < 420, (len(sms), sms)
    title, body = ntfy_notifier.build_message("AC1", False, a["lines"], row, action_guide.short_actions(a["findings"], ml, 2, a), "ml", a)
    assert title == "Check (ML): AC1"
    for needle in ("Critical", "Operating current", "usual", "Risk:", "Why:", "Do this now:"):
        assert needle in body, (needle, body)


def test_ml_alert_json_has_assessment_condition_and_no_good_condition_trigger():
    ml = SIM_DET.check("SIM", SCEN["sim_fault (over every limit)"][0])
    saved, summary, sms = _run_process_anomaly("ml", SCEN["sim_fault (over every limit)"][0], ml)
    assert saved["ml_assessment"]["severity"] == "Critical" and saved["ml_assessment"]["condition"] == "Need Repair"
    assert saved["rule_diagnoses"]["overall_severity"] == "Critical" and saved["rule_diagnoses"]["sensors_triggered"] >= 4
    assert saved["action_plan"]["steps"] and saved["action_plan"]["urgency"] == "stop_now"
    assert "normal" not in summary.lower().split("[ml]")[0], "the old keyword trap ('normal' = healthy) must not be possible"


# ---- 3. ntfy against a local fake server --------------------------------------------------------------
import http.server
import threading


class _Fake(http.server.BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Fake.seen.append({"path": self.path, "headers": dict(self.headers), "body": body.decode("utf-8")})
        code = {"ok": 200, "ratelimit": 429, "denied": 403, "strict": 200, "badtoken": 200}[_Fake.mode]
        if _Fake.mode == "badtoken" and "Authorization" in self.headers:    # a public topic that rejects a wrong token
            code = 401
        if _Fake.mode == "strict" and "Click" in self.headers:      # a server that rejects an odd optional header
            code = 400
        msg = b'{"error":"unauthorized"}' if code == 401 else b'{"error":"limit reached"}' if code not in (200, 400) else (b'{"error":"invalid click url"}' if code == 400 else b'{"id":"x"}')
        self.send_response(code)
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def log_message(self, *a): pass


def _ntfy_env(port, topic="test_topic_k8x2", **extra):
    for k in ("NTFY_TOPIC", "NTFY_SERVER", "NTFY_TOKEN", "APP_URL"):
        os.environ.pop(k, None)
    os.environ["NTFY_TOPIC"] = topic
    os.environ["NTFY_SERVER"] = f"http://127.0.0.1:{port}"
    os.environ.update(extra)
    ntfy_notifier.RETRY_WAITS = (0, 0.01, 0.01)
    _Fake.seen.clear()
    for k in ("sent", "failed"): ntfy_notifier._status[k] = 0


def _with_server(fn):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        fn(srv.server_address[1])
    finally:
        srv.shutdown()


def test_ntfy_delivers_ml_and_safety_alerts_with_the_right_request():
    def run(port):
        _Fake.mode = "ok"
        _ntfy_env(port, topic='  "test_topic_k8x2"  ', APP_URL="https://x.example", NTFY_TOKEN="tk_abc")   # sloppy env values
        assert ntfy_notifier.send_alert("AC1", False, ["ML model: x"], source="ml") is True
        assert ntfy_notifier.send_alert("AC1", True, ["Current is 12 A"], source="safety") is True
        a, b = _Fake.seen
        assert a["path"] == "/test_topic_k8x2" and a["headers"]["Title"] == "Check (ML): AC1" and a["headers"]["Priority"] == "4"
        assert b["headers"]["Title"] == "STOP (Safety): AC1" and b["headers"]["Priority"] == "5"
        assert a["headers"]["Authorization"] == "Bearer tk_abc" and a["headers"]["Click"] == "https://x.example/pages/alerts"
        assert a["headers"]["User-Agent"].startswith("AirBnBrrr")
        st = ntfy_notifier.status()
        assert st["last_ok"] is True and st["sent"] == 2 and st["failed"] == 0 and st["configured"]
    _with_server(run)


def test_ntfy_rate_limit_is_retried_then_reported_with_the_reason():
    def run(port):
        _Fake.mode = "ratelimit"
        _ntfy_env(port)
        assert ntfy_notifier.send_alert("AC1", False, ["x"], source="ml") is False
        assert len(_Fake.seen) == 3, "429 is retried"
        st = ntfy_notifier.status()
        assert st["last_ok"] is False and st["last_http_status"] == 429 and "rate limited" in st["last_error"] and st["failed"] == 1
    _with_server(run)


def test_ntfy_auth_failure_is_not_retried_and_says_what_to_fix():
    def run(port):
        _Fake.mode = "denied"
        _ntfy_env(port)
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is False
        assert len(_Fake.seen) == 1
        assert "NTFY_TOKEN" in ntfy_notifier.status()["last_error"]
    _with_server(run)


def test_ntfy_still_delivers_when_the_server_rejects_an_optional_header():
    def run(port):
        _Fake.mode = "strict"
        _ntfy_env(port, APP_URL="onrender-app.example")             # bare APP_URL, no https://
        assert ntfy_notifier.send_alert("AC1", True, ["Current is 12 A"], source="safety") is True
        first = _Fake.seen[0]["headers"]
        assert first["Click"] == "https://onrender-app.example/pages/alerts", "APP_URL is repaired"
        # now a server that rejects ANY Click header: the retry must drop it and still deliver
        _Fake.seen.clear(); os.environ["APP_URL"] = "https://ok.example"
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="ml") is True
        assert len(_Fake.seen) == 2 and "Click" in _Fake.seen[0]["headers"] and "Click" not in _Fake.seen[1]["headers"]
        assert _Fake.seen[1]["headers"]["Title"] == "STOP (ML): AC1"
        assert ntfy_notifier.status()["last_ok"] is True
    _with_server(run)


def test_ntfy_server_address_without_scheme_gets_https():
    os.environ["NTFY_SERVER"] = "ntfy.sh"
    assert ntfy_notifier._server() == "https://ntfy.sh"
    os.environ["NTFY_SERVER"] = ' "http://my.server:8080/" '
    assert ntfy_notifier._server() == "http://my.server:8080"
    os.environ.pop("NTFY_SERVER")
    assert ntfy_notifier._server() == "https://ntfy.sh"


def test_ntfy_unreachable_and_missing_topic_are_reported():
    _ntfy_env(1)                                             # nothing listens on port 1
    assert ntfy_notifier.send_alert("AC1", False, ["x"]) is False
    assert ntfy_notifier.status()["last_ok"] is False and ntfy_notifier.status()["last_error"]
    os.environ.pop("NTFY_TOPIC")
    assert ntfy_notifier.send_alert("AC1", False, ["x"]) is False
    st = ntfy_notifier.status()
    assert st["configured"] is False and "NTFY_TOPIC is not set" in st["last_error"]


# ---- 4. instant alerts ---------------------------------------------------------------------------------
def test_instant_alert_is_not_held_back_by_a_recent_normal_alert():
    """The bug from the screenshots: a normal (3-minute) safety alert had just been sent, then a real fault began."""
    sch = ch.AlertScheduler()
    got = {}
    for i in range(0, 60):                       # 10 minutes at 10 s
        t = i * 10
        fault = t >= 250                         # real fault from t=250 s
        soft_problem = True                      # e.g. vibration limit passed all the time
        a = sch.decide("AC1", "safety", t, fault, soft_problem, {"payload": {}})
        if a: got.setdefault(a, []).append(t)
    assert got["soft"][0] == 180, got           # the normal alert after its 3 minutes
    assert got["critical"][0] == 250, f"instant alert must come at the very first fault reading, not after a pause: {got}"


def test_instant_alert_needs_no_confirmation_time_and_normal_alerts_keep_theirs():
    sch = ch.AlertScheduler()
    assert sch.decide("AC1", "safety", 0, True, False, {"payload": {}}) == "critical"          # first reading
    sch2 = ch.AlertScheduler()
    assert [sch2.decide("AC1", "ml", t, False, True, {"payload": {}}) for t in range(0, 180, 10)] == [None] * 18
    assert sch2.decide("AC1", "ml", 180, False, True, {"payload": {}}) == "soft"


def test_compressor_limit_plus_model_is_instant_and_the_model_can_only_speed_it_up():
    base = dict(SCEN["sim_drift_peak (inside limits)"][5], vibration=95.0)           # vibration over the limit (50)
    L = limits.merged(None)
    rules = rule_engine.analyze_with_rules(base, L)
    ml_bad = {"is_anomaly": True, "margin": -0.5, "distance_ratio": 6.0, "checks": ["distance"], "operating_mode": "running",
              "deviations": [{"key": "vibration", "value": 95.0, "mid": 52.0, "low": 40.0, "high": 64.0, "z": 12.0, "direction": "above"}]}
    on_agree = ch.evaluate_channels(dict(base), L, ml_bad, True, rules, ntfy_notifier.build_problems)
    alone = ch.evaluate_channels(dict(base), L, None, True, rules, ntfy_notifier.build_problems)
    off = ch.evaluate_channels(dict(base), L, ml_bad, False, rules, ntfy_notifier.build_problems)
    assert on_agree["safety"]["crit"] is True, "limit passed AND the model agrees: INSTANT"
    assert alone["safety"]["crit"] is False and alone["safety"]["soft"] is True, "limit alone: normal 3-minute alert, never lost"
    assert off["safety"]["crit"] is False and off["safety"]["soft"] is False and off["ml"]["soft"], "switch off: ML alert unchanged"


def test_alert_is_saved_and_sms_sent_before_the_slow_ai_step():
    ml = SIM_DET.check("SIM", SCEN["sim_fault (over every limit)"][0])
    _run_process_anomaly("safety", dict(SCEN["sim_fault (over every limit)"][0]), ml)
    assert CAPTURED["events"] == ["save", "sms", "ask", "update"], CAPTURED["events"]
    first = json.loads(CAPTURED["alerts"][0].diagnoses)
    assert first["ai_diagnoses"].get("pending") is True, "the alert is complete and visible before the AI answers"
    assert "error" in json.loads(CAPTURED["updates"][0][1])["ai_diagnoses"]            # AI failed in the test: reported, not lost
    assert CAPTURED["sms"][0].rstrip().endswith("Report ID: 1")


def test_legacy_humidity_low_of_zero_is_replaced_by_the_default():
    assert limits.merged({"humidity": {"min": 0, "max": 80}})["humidity"] == {"min": 30.0, "max": 100.0}
    assert limits.merged({"humidity": {"min": 25, "max": 100}})["humidity"]["min"] == 25.0


# =====================================================================================================
# v3.3: instant ML alerts, new-episode timer, hot path off the database
# =====================================================================================================
def _instant_rate(rows):
    n = 0
    for r in rows:
        ml = SIM_DET.check("SIM", r)
        a = ml_explain.assess(ml, r, "SIM") if ml["is_anomaly"] else None
        n += bool(a and a["instant"])
    return n / len(rows)


def test_ml_instant_only_for_severe_findings_and_never_on_normal_data():
    fresh = fault_data.normal_rows(3000, seed=123, start=2e6).to_dict("records")
    assert _instant_rate(fresh) == 0.0, "normal readings must never be an instant ML alert"
    for name in ("sim_fault (over every limit)", "overvoltage 250 V", "refrigerant leak", "compressor stuck off while cooling"):
        assert _instant_rate(SCEN[name]) >= 0.95, name
    for name in ("sim_drift_peak (inside limits)", "sim_drift_half (inside limits)", "undervoltage 205 V",
                 "frozen coil / clogged filter", "bearing wear (vibration only)", "outlet compressor too hot (one sensor)"):
        assert _instant_rate(SCEN[name]) == 0.0, f"{name} keeps the normal 3-minute confirmation"


def test_instant_ml_alert_works_with_the_safety_switch_off():
    row = SCEN["sim_fault (over every limit)"][2]
    ml = SIM_DET.check("SIM", row)
    L = limits.merged(None)
    off = ch.evaluate_channels(dict(row), L, ml, False, rule_engine.analyze_with_rules(row, L), ntfy_notifier.build_problems, "AC1")
    assert off["ml"]["crit"] is True and off["ml"]["snapshot"]["assessment"]["instant"]
    assert off["safety"]["crit"] is False and off["safety"]["soft"] is False


def test_power_factor_has_no_safety_limit_so_it_can_only_come_from_the_model():
    assert "power_factor" not in {m["key"] for m in limits.LIMIT_META}
    assert not any("power_factor" in r["id"] for r in rule_engine.build_rules(limits.merged(None)))


def test_a_new_instant_problem_is_not_held_back_but_flapping_is_limited():
    sch = ch.AlertScheduler()
    assert sch.decide("A", "safety", 0, True, False, {"payload": {}}) == "critical"
    got = [sch.decide("A", "safety", t, False, False) for t in range(10, 90, 10)]          # 80 s normal: problem over
    assert not any(got)
    assert sch.decide("A", "safety", 90, True, False, {"payload": {}}) == "critical", "a new fault 90 s later alerts at once"
    flap = ch.AlertScheduler()
    assert flap.decide("A", "safety", 0, True, False, {"payload": {}}) == "critical"
    flap.decide("A", "safety", 10, False, False)
    flap.decide("A", "safety", 20, False, False)                                   # most of the last readings are normal: episode over
    assert flap.decide("A", "safety", 25, True, False, {"payload": {}}) is None, "back inside 30 s: flapping is still limited"
    assert flap.decide("A", "safety", 30, True, False, {"payload": {}}) == "critical", "a new problem 30 s after the last alert goes out"


def test_limits_and_settings_never_block_on_the_database_once_read():
    import threading as th
    calls = {"n": 0}

    class C:
        def cursor(self): return self
        def execute(self, *a, **k): pass
        def fetchone(self): return None
        def fetchall(self): return []
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    def slow():
        calls["n"] += 1
        time.sleep(0.5)
        return C()
    limits._cache.clear()
    limits.get(slow, "ZZ", ttl=0.0)                      # first read: may wait
    t0 = time.time()
    got = limits.get(slow, "ZZ", ttl=0.0)                # stale: must answer at once and refresh in the background
    assert time.time() - t0 < 0.1 and got["current"]["max"] > 0
    app_settings.invalidate()
    app_settings.get_all(slow, ttl=5)
    t0 = time.time()
    app_settings.get_all(slow, ttl=0.0001)
    assert time.time() - t0 < 0.1


def test_legacy_model_without_distance_check_is_never_critical_or_instant():
    """Real stored models trained before the distance check existed graded 67 of 72 ordinary outliers 'Critical' by z-score alone."""
    legacy = {k: v for k, v in SIM_BUNDLE.items() if k != "knn_ref"}
    for name in ("sim_fault (over every limit)", "compressor stuck off while cooling", "refrigerant leak"):
        for row in SCEN[name][:10]:
            ml = ml_per_ac.score_reading(legacy, row)
            assert ml["distance_ratio"] is None
            if ml["is_anomaly"]:
                a = ml_explain.assess(ml, row, "AC6")
                assert a["severity"] in ("Low", "Medium") and a["instant"] is None and a["legacy_model"] is True
                assert "Train now" in a["why"]
                assert a["condition"] == "Need Maintenance"


def test_a_severe_ml_finding_needs_two_readings_in_a_row_but_the_safety_instant_needs_one():
    sch = ch.AlertScheduler()
    assert sch.decide("A", "ml", 0, True, True, {"payload": {}}) is None            # 1st severe reading: not yet
    assert sch.decide("A", "ml", 10, True, True, {"payload": {}}) == "critical"     # 2nd in a row: INSTANT
    glitch = ch.AlertScheduler()
    assert glitch.decide("A", "ml", 0, True, True, {"payload": {}}) is None
    assert glitch.decide("A", "ml", 10, False, False) is None                       # back to normal: streak broken
    assert glitch.decide("A", "ml", 20, True, True, {"payload": {}}) is None
    assert ch.AlertScheduler().decide("A", "safety", 0, True, False, {"payload": {}}) == "critical"


# =====================================================================================================
# v3.5: ntfy when the server has no network ("Errno 101: Network is unreachable")
# =====================================================================================================
import errno
import socket as _socket


def _net_reset():
    ntfy_notifier._getaddrinfo = _socket.getaddrinfo
    ntfy_notifier._make_socket = _socket.socket
    with ntfy_notifier._pending_lock:
        ntfy_notifier._pending.clear()
    ntfy_notifier.RETRY_QUEUE_DELAYS = (0.05, 0.05)
    ntfy_notifier.WORKER_TICK = 0.03
    ntfy_notifier.RETRY_WAITS = (0, 0.01, 0.01)


def _no_route_socket(*a, **k):
    class S:
        def settimeout(self, *x): pass
        def connect(self, *x): raise OSError(errno.ENETUNREACH, "Network is unreachable")
        def close(self): pass
    return S()


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_ntfy_tries_ipv4_before_ipv6_and_survives_a_missing_ipv6_route():
    def run(port):
        _Fake.mode = "ok"; _ntfy_env(port); _net_reset()
        tried = []
        real = _socket.socket
        def factory(fam, typ, proto=0):
            tried.append(fam)
            if fam == _socket.AF_INET6:
                return _no_route_socket()
            return real(fam, typ, proto)
        ntfy_notifier._getaddrinfo = lambda h, p, f=0, t=0: [(_socket.AF_INET6, t, 6, "", ("::1", p, 0, 0)), (_socket.AF_INET, t, 6, "", ("127.0.0.1", p))]
        ntfy_notifier._make_socket = factory
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is True
        assert tried[0] == _socket.AF_INET, "IPv4 must be tried first"
        _net_reset()
    _with_server(run)


def test_no_network_is_reported_clearly_then_the_alert_is_delivered_when_it_returns():
    def run(port):
        _Fake.mode = "ok"; _ntfy_env(port); _net_reset()
        ntfy_notifier._make_socket = _no_route_socket                      # every address: Errno 101
        before = ntfy_notifier.status()["failed"]
        assert ntfy_notifier.send_alert("AC1", True, ["Current is 12 A"], source="safety") is False
        st = ntfy_notifier.status()
        assert st["last_ok"] is False and st["failed"] == before + 1
        assert "No network route" in st["last_error"] and "Errno 101" in st["last_error"] and "not a topic or setting problem" in st["last_error"]
        assert "127.0.0.1" in st["last_error"], "the error lists every address that was tried"
        assert st["queued"] == 1 and not _Fake.seen, "nothing was delivered, the alert is kept"
        ntfy_notifier._make_socket = _socket.socket                          # the network comes back
        assert _wait(lambda: len(_Fake.seen) == 1), "the queued alert must be delivered automatically"
        assert _Fake.seen[0]["headers"]["Title"] == "STOP (Safety): AC1"
        assert _wait(lambda: ntfy_notifier.status()["queued"] == 0)
        assert ntfy_notifier.status()["last_ok"] is True
        _net_reset()
    _with_server(run)


def test_a_test_notification_is_never_queued_and_a_permanent_outage_is_given_up_on():
    def run(port):
        _Fake.mode = "ok"; _ntfy_env(port); _net_reset()
        ntfy_notifier._make_socket = _no_route_socket
        assert ntfy_notifier.send_test() is False
        assert ntfy_notifier.status()["queued"] == 0, "a test reports the failure at once instead of hiding it in a queue"
        ntfy_notifier.send_alert("AC1", False, ["x"], source="ml")
        assert _wait(lambda: ntfy_notifier.status()["queued"] == 0, timeout=4), "after the retries are used up the alert is dropped, not kept forever"
        assert not _Fake.seen
        _net_reset()
    _with_server(run)


def test_http_errors_are_not_queued_only_missing_network_is():
    def run(port):
        _Fake.mode = "denied"; _ntfy_env(port); _net_reset()
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is False
        assert ntfy_notifier.status()["queued"] == 0, "a 403 will not fix itself: it is reported, not retried for minutes"
        _net_reset()
    _with_server(run)


def test_network_diagnosis_names_the_cause():
    os.environ.pop("NTFY_SERVER", None)
    _net_reset()
    # DNS down
    def dns_fail(*a, **k): raise OSError(-2, "Name or service not known")
    ntfy_notifier._getaddrinfo = dns_fail
    d = ntfy_notifier.diagnose_network()
    assert d["ok"] is False and "cannot look up" in d["verdict"]
    # DNS fine, no route on any address
    ntfy_notifier._getaddrinfo = lambda h, p, f=0, t=0: [(_socket.AF_INET, t, 6, "", ("203.0.113.5", p)), (_socket.AF_INET6, t, 6, "", ("2001:db8::5", p, 0, 0))]
    ntfy_notifier._make_socket = _no_route_socket
    d = ntfy_notifier.diagnose_network()
    assert d["ok"] is False and "NOT reachable" in d["verdict"] and any("Network is unreachable" in l for l in d["lines"])
    # IPv4 works, IPv6 has no route: harmless
    real = _socket.socket
    def factory(fam, typ, proto=0):
        if fam == _socket.AF_INET6: return _no_route_socket()
        class OK:
            def settimeout(self, *x): pass
            def connect(self, *x): pass
            def close(self): pass
        return OK()
    ntfy_notifier._make_socket = factory
    d = ntfy_notifier.diagnose_network()
    assert d["ok"] is True and "IPv6 has no route" in d["verdict"] and "harmless" in d["verdict"]
    _net_reset()


# =====================================================================================================
# v3.5: advice comes from the COMBINATION of odd sensors (probable causes), not canned text per sensor
# =====================================================================================================
def _plans(name, n=20):
    out = []
    for r in SCEN[name][:n]:
        ml = SIM_DET.check("SIM", r)
        if ml["is_anomaly"]:
            out.append(ml_explain.assess(ml, r, "AC1"))
    return out


def _primary(a):
    return a["steps"][0]["id"].replace("ml_", "") if a["steps"] else None


def test_the_leading_cause_matches_the_fault_that_was_injected():
    expected = {
        "sim_fault (over every limit)": "overvoltage",
        "overvoltage 250 V": "overvoltage",
        "undervoltage 205 V": "undervoltage",
        "refrigerant leak": "low_refrigerant",          # NOT "compressor not running": the outlet pipe is hot, so it runs
        "frozen coil / clogged filter": "airflow_icing",
        "bearing wear (vibration only)": "vibration",
        "compressor stuck off while cooling": "compressor_not_running",
        "outlet compressor too hot (one sensor)": "compressor_heat",
    }
    for name, want in expected.items():
        plans = _plans(name)
        assert plans, name
        got = collections.Counter(_primary(a) for a in plans).most_common(1)[0][0]
        assert got == want, (name, got, want)


def test_a_plan_never_repeats_an_action_and_does_not_spam_the_filter():
    n = 0
    for name in SCEN:
        for a in _plans(name):
            n += 1
            acts = [x for s in a["steps"] for x in s["do_now"] if x != "Same checks as the steps above."]
            assert len(acts) == len(set(acts)), (name, acts)
            assert sum("filter" in x.lower() for x in acts) <= 1, (name, [x for x in acts if "filter" in x.lower()])
            assert len({s["id"] for s in a["steps"]}) == len(a["steps"])
    assert n > 100


def test_a_supply_voltage_problem_is_sent_to_the_electrician_not_the_air_filter():
    for a in _plans("overvoltage 250 V") + _plans("sim_fault (over every limit)"):
        first = a["steps"][0]
        assert first["id"] == "ml_overvoltage" and "electrician" in " ".join(first["do_now"]).lower()
        assert "filter" not in " ".join(first["do_now"]).lower()
        assert first["because"] and "V" in first["because"], "the readings that support the cause are shown"
        assert first["confidence"] in ("Most likely cause", "Likely cause")


def test_every_odd_sensor_is_covered_by_a_cause_or_listed_on_its_own():
    for name in SCEN:
        for a in _plans(name, 8):
            covered = {k for s in a["steps"] for k in s.get("sensors", [])}
            assert {o["key"] for o in a["odd"]} <= covered or not a["odd"], (name, {o["key"] for o in a["odd"]} - covered)


def test_ml_finding_value_is_a_plain_number_so_the_gauge_can_judge_it():
    """The page compared the reading as a number: text with the unit made it say 'Normal' and print the unit twice."""
    for a in _plans("sim_fault (over every limit)", 5):
        for f in a["findings"]:
            if f["sensor_key"] == "ml_pattern":
                continue
            assert float(f["measured_value"]) == float(f["measured_value"]), f
            assert f["unit"] not in f["measured_value"], f
            assert "–" in f["normal_range"] and f["unit"] in f["normal_range"]


def test_each_cause_step_has_what_the_page_renders():
    for a in _plans("sim_fault (over every limit)", 3):
        for s in a["steps"]:
            for k in ("title", "severity", "urgency", "short", "why", "do_now", "check", "call_technician_if", "confidence"):
                assert k in s, (s["id"], k)
        assert a["causes"] and a["causes"][0]["id"] == "overvoltage"
        assert "most likely cause" in a["why"].lower()


# ---- NTFY_TOKEN handling (v3.5) ----
def test_a_rejected_token_does_not_stop_alerts_on_a_public_topic_but_is_reported():
    def run(port):
        _Fake.mode = "badtoken"; _ntfy_env(port, NTFY_TOKEN="not-a-real-token"); _net_reset()
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is True, "the alert must still arrive"
        assert len(_Fake.seen) == 2 and "Authorization" in _Fake.seen[0]["headers"] and "Authorization" not in _Fake.seen[1]["headers"]
        st = ntfy_notifier.status()
        assert st["last_ok"] is True and st["warning"] and "NTFY_TOKEN" in st["warning"] and "tk_" in st["warning"] and st["token_set"] is True
        _Fake.mode = "ok"; _Fake.seen.clear()
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is True
        _net_reset()
    _with_server(run)


def test_a_protected_topic_with_a_bad_token_says_what_to_check_not_to_set_the_token():
    def run(port):
        _Fake.mode = "denied"; _ntfy_env(port, NTFY_TOKEN="tk_wrong"); _net_reset()
        assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is False
        err = ntfy_notifier.status()["last_error"]
        assert "rejected" in err and "tk_" in err and "SAME ntfy server" in err and "set NTFY_TOKEN" not in err
        os.environ.pop("NTFY_TOKEN")
        _Fake.seen.clear(); assert ntfy_notifier.send_alert("AC1", True, ["x"], source="safety") is False
        assert "set NTFY_TOKEN" in ntfy_notifier.status()["last_error"], "with no token set, the advice is to set one"
        _net_reset()
    _with_server(run)


def test_token_formats_people_paste_are_understood():
    import base64
    for raw, expect in (("tk_abc123", "Bearer tk_abc123"), ('  "tk_abc123" ', "Bearer tk_abc123"), ("Bearer tk_abc123", "Bearer tk_abc123"),
                        ("me:secret", "Basic " + base64.b64encode(b"me:secret").decode())):
        os.environ["NTFY_TOKEN"] = raw
        assert ntfy_notifier._auth_header() == expect, raw
    os.environ["NTFY_TOKEN"] = ""
    assert ntfy_notifier._auth_header() is None
    os.environ.pop("NTFY_TOKEN")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:
            failed += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
