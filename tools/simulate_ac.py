#!/usr/bin/env python3
"""
Simulate one AC sending live readings to your AirBnBrrr site, so you can watch the model and the alerts work.

    python tools/simulate_ac.py            (sends to the site address in DEFAULT_URL below, as AC7)
    python tools/simulate_ac.py --url https://other-address.onrender.com --unit AC7

It needs only Python 3 (nothing to install). Before you start:
  1. The AC must have a trained model. AC7 is the "Demo AC" with realistic training data: Manage ACs -> Demo AC -> Train now.
  2. AC Setup must be in MONITORING (telemetry) mode, not "data gathering": otherwise the fake readings
     would be saved as training data. The script checks this and refuses to run if it is wrong.
  3. If you set TELEMETRY_API_KEY on Render, pass it with --key (or set the same environment variable).

The default "demo" story takes about 30 minutes (the website waits 8 minutes before a basic alert, so it cannot be
faster). While it runs, open Model Health and Alerts, and keep the ntfy app on your phone.

    minutes  0-4   normal readings              -> the AC looks normal on Model Health
    minutes  4-9   a blocked water drain: humidity and the air temperature creep up, all still INSIDE their limits
    minutes  9-23  it stays that way. No instant safety limit is broken, so it is the AI model that notices.
                   About 8 minutes after it does, a "Check: ..." alert arrives (ntfy)
    minutes 23-25  a real fault: 12.5 A, 250 V...   -> instant "STOP: ..." alert
    minutes 25-29  back to normal

Other stories: --scenario normal | drift | fault | offline
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

DEFAULT_URL = "https://airrrr.onrender.com"   # change this if your Render address changes again

SCENARIOS = {   # (phase, minutes)
    "demo":    [("normal", 4), ("drift_up", 5), ("drift_hold", 14), ("fault", 2), ("recover", 4)],
    "normal":  [("normal", 10)],
    "drift":   [("normal", 3), ("drift_up", 5), ("drift_hold", 12)],
    "fault":   [("normal", 1), ("fault", 2), ("recover", 3)],
    "offline": [("normal", 1), ("offline", 3), ("recover", 2)],
}
NOTES = {
    "normal": "normal readings",
    "drift_up": "blocked water drain: humidity and temperatures creeping up",
    "drift_hold": "humidity stays high, the AI model should notice",
    "fault": "REAL FAULT: over the limits, instant alert expected",
    "offline": "power meter stops answering",
    "recover": "back to normal",
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def reading(t, rng, drift=0.0, fault=False, offline=False):
    """One reading `t` seconds into the story. Normal behaviour is a steady cooling AC (same formulas as the
    training rows of the Demo AC). Drift is a blocked water drain: humidity and the air temperatures creep up,
    but every value stays INSIDE its safety limit. A fault breaks the limits."""
    i = t / 18.0
    u = lambda: rng.random() - 0.5
    w = math.sin(i / 160)                                   # slow load cycle shared by several sensors
    cur = 8.15 + 0.12 * w + u() * 0.12
    volt = 232 + 1.2 * math.sin(i / 300 + 1) + u() * 1.0
    pf = 0.945 + u() * 0.008
    r = {"dht_temp": 14.0 + 0.8 * w + u() * 0.6,
         "dht_humidity": 62 + 2.5 * math.sin(i / 220 + 2) + u() * 2.0,
         "vibration": 52 + 5 * w + u() * 6,
         "ds18b20_temp1": 55.5 + 0.9 * w + u() * 0.6,
         "ds18b20_temp2": 12.5 + 0.7 * w + u() * 0.5,
         "pzem_voltage": volt, "pzem_current": cur, "pzem_power": volt * cur * pf,
         "pzem_frequency": 60 + u() * 0.06, "pzem_power_factor": pf}
    if drift > 0:   # blocked water drain: nothing crosses a limit (every value is capped below its limit)
        r["dht_humidity"] = min(79, r["dht_humidity"] + (78 - 62) * drift)
        r["dht_temp"] = min(24, r["dht_temp"] + (20 - 14) * drift)
        r["ds18b20_temp2"] = min(16.5, r["ds18b20_temp2"] + (15.5 - 12.5) * drift)
        r["ds18b20_temp1"] = min(60, r["ds18b20_temp1"] + (59 - 55.5) * drift)
        r["pzem_current"] = min(8.7, r["pzem_current"] + 0.2 * drift)
        r["pzem_power"] = min(1850, volt * r["pzem_current"] * pf)
        r["vibration"] = min(85, r["vibration"] + 14 * drift)
    if fault:
        r.update({"pzem_current": 12.5 + u(), "pzem_voltage": 250 + u() * 2, "pzem_power": 2900 + u() * 100, "vibration": 105 + u() * 6,
                  "ds18b20_temp1": 72 + u() * 2, "ds18b20_temp2": 28 + u() * 2, "dht_temp": 38 + u()})
    if offline:
        for k in ("pzem_voltage", "pzem_current", "pzem_power", "pzem_frequency", "pzem_power_factor"):
            r[k] = None
    return {k: (round(v, 2) if v is not None else None) for k, v in r.items()}


def timeline(scenario, interval):
    """[(phase, virtual seconds since start, drift 0..1)] for the whole story."""
    steps, t, drift = [], 0.0, 0.0
    for phase, minutes in SCENARIOS[scenario]:
        n = int(minutes * 60 / interval)
        for k in range(n):
            if phase == "drift_up":
                drift = (k + 1) / n
            elif phase in ("normal", "recover", "fault", "offline"):
                drift = 0.0 if phase != "recover" else max(0.0, drift)
            steps.append((phase, t, drift if phase in ("drift_up", "drift_hold") else 0.0))
            t += interval
    return steps


def http(method, url, payload=None, key=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
        return resp.status, (json.loads(body) if body else None)


def main():
    ap = argparse.ArgumentParser(description="Simulate live readings of one AC.")
    ap.add_argument("--url", default=os.environ.get("AIRBNBRRR_URL", DEFAULT_URL), help="your site address (default: " + DEFAULT_URL + ")")
    ap.add_argument("--unit", default="AC7", help="device code of the AC to simulate (it needs a trained model)")
    ap.add_argument("--key", default=os.environ.get("TELEMETRY_API_KEY", ""), help="the TELEMETRY_API_KEY you set on Render, if any")
    ap.add_argument("--scenario", default="demo", choices=sorted(SCENARIOS))
    ap.add_argument("--interval", type=float, default=10, help="seconds between readings (like the ESP32)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="testing only: 0.01 waits 1%% of the real time")
    ap.add_argument("--force", action="store_true", help="run even if AC Setup is in data gathering mode (it will save fake training data!)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if not a.url:
        sys.exit("Give your site address with --url, for example --url https://airrrr.onrender.com")
    base = a.url.rstrip("/")

    try:   # safety check: do not turn the demo into training data
        _, setup = http("GET", base + "/ac_setup", timeout=90)
    except Exception as e:
        sys.exit(f"Could not reach {base}: {e}\n(Render's free plan can take a minute to wake up; try again.)")
    if (setup or {}).get("data_gathering_mode") == "data_gathered" and not a.force:
        sys.exit("AC Setup is in DATA GATHERING mode: these fake readings would be saved as training data for "
                 f"{(setup or {}).get('data_gathering_unit')}.\nOpen AC Setup, switch to monitoring (telemetry) mode, then run this again.")

    steps = timeline(a.scenario, a.interval)
    rng = random.Random(a.seed)
    started = datetime.now(timezone.utc)
    print(f"Simulating {a.unit}: '{a.scenario}' story, {len(steps)} readings, about {len(steps) * a.interval / 60:.0f} minutes. Ctrl+C stops it.\n")
    last_phase, energy, sent = None, 40000.0, 0
    try:
        for phase, t, drift in steps:
            if phase != last_phase:
                print(f"\n== {t / 60:4.0f} min  {phase.upper():10s} {NOTES[phase]}")
                last_phase = phase
            r = reading(t, rng, drift=drift, fault=(phase == "fault"), offline=(phase == "offline"))
            if r["pzem_power"] is not None:
                energy += r["pzem_power"] * a.interval / 3600.0
                r["pzem_energy"] = round(energy, 1)
            else:
                r["pzem_energy"] = None
            r["ac_unit"] = a.unit
            try:
                status, _ = http("POST", base + "/telemetry", r, a.key)
                note = "" if status in (200, 201) else f" (server said {status})"
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    sys.exit("\nThe site refused the readings (401): pass the same text you set as TELEMETRY_API_KEY with --key.")
                note = f" (error {e.code})"
            except Exception as e:
                note = f" (could not send: {e})"
            sent += 1
            cur, volt, t1 = r["pzem_current"], r["pzem_voltage"], r["ds18b20_temp1"]
            print(f"  {t / 60:5.1f} min  {cur if cur is not None else '--':>6} A  {volt if volt is not None else '--':>6} V  outlet {t1:>5} C  vib {r['vibration']:>6}{note}")
            time.sleep(a.interval * a.time_scale)
    except KeyboardInterrupt:
        print("\nStopped.")
    ended = datetime.now(timezone.utc)
    s0, s1 = (started - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S+00"), (ended + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S+00")
    print(f"\nDone: {sent} readings sent. To remove the fake data afterwards, run this in the Supabase SQL editor:\n")
    print(f"  delete from telemetry where timestamp >= '{s0}' and timestamp <= '{s1}';")
    print(f"  delete from ac_health where ac_unit = '{a.unit}' and ts >= '{s0}' and ts <= '{s1}';")
    print(f"  delete from alerts where timestamp >= '{s0}' and timestamp <= '{s1}';   -- or delete them on the Alerts page")


if __name__ == "__main__":
    main()
