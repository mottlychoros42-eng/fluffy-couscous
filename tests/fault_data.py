"""
Abnormal-AC datasets for testing the model with real inference.

NORMAL behaviour comes from the project's own simulator (tools/simulate_ac.py, the same formulas as the Demo AC's
training rows). The FAULTS are physical failure patterns of a window AC. This is generated data: your own
`Export CSV` of an AC (Manage ACs) can be tested the same way with tools/check_model.py.
"""
import os
import random
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import simulate_ac as sim  # noqa: E402


def normal_rows(n=4000, seed=1, step=10.0, start=0.0):
    rng = random.Random(seed)
    return pd.DataFrame([sim.reading(start + i * step, rng) for i in range(n)])


def _base(seed, n=60):
    rng = random.Random(seed)
    return [sim.reading(900000 + i * 10.0, rng) for i in range(n)]


def _apply(rows, fn):
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        fn(r, i)
        out.append(r)
    return out


def _power(r):
    r["pzem_power"] = round(r["pzem_voltage"] * r["pzem_current"] * r["pzem_power_factor"], 2)


def scenarios(n=60):
    """name -> list of readings. Each has the kind of severity a technician would expect."""
    rng = random.Random(7)
    S = {}
    S["sim_fault (over every limit)"] = [sim.reading(5000 + i * 10.0, rng, fault=True) for i in range(n)]
    S["sim_drift_peak (inside limits)"] = [sim.reading(5000 + i * 10.0, rng, drift=1.0) for i in range(n)]
    S["sim_drift_half (inside limits)"] = [sim.reading(5000 + i * 10.0, rng, drift=0.5) for i in range(n)]

    def overvolt(r, i): r["pzem_voltage"] = round(250 + rng.random(), 2); _power(r)
    S["overvoltage 250 V"] = _apply(_base(11, n), overvolt)

    def undervolt(r, i):
        r["pzem_voltage"] = round(205 + rng.random(), 2); r["pzem_current"] = round(r["pzem_current"] * 1.12, 2); _power(r)
    S["undervoltage 205 V"] = _apply(_base(12, n), undervolt)

    def leak(r, i):
        r.update(pzem_current=round(6.6 + rng.random() * .2, 2), dht_temp=round(27 + rng.random(), 2), ds18b20_temp1=round(71 + rng.random(), 2),
                 ds18b20_temp2=round(21 + rng.random(), 2)); _power(r)
    S["refrigerant leak"] = _apply(_base(13, n), leak)

    def freeze(r, i):
        r.update(dht_temp=round(5 + rng.random(), 2), ds18b20_temp2=round(3 + rng.random(), 2), pzem_current=round(7.3 + rng.random() * .1, 2)); _power(r)
    S["frozen coil / clogged filter"] = _apply(_base(14, n), freeze)

    def bearing(r, i): r["vibration"] = round(95 + rng.random() * 8, 2)
    S["bearing wear (vibration only)"] = _apply(_base(15, n), bearing)

    def stuck(r, i):
        r.update(pzem_current=round(0.2 + rng.random() * .05, 2), pzem_power=round(18 + rng.random() * 3, 2), vibration=round(3 + rng.random(), 2),
                 pzem_power_factor=0.45)
    S["compressor stuck off while cooling"] = _apply(_base(16, n), stuck)

    def hot_only(r, i): r["ds18b20_temp1"] = round(66 + rng.random(), 2)
    S["outlet compressor too hot (one sensor)"] = _apply(_base(17, n), hot_only)
    return S
