"""
Independent alert channels for AirBnBrrr (v3.0).

Every incoming reading is judged by THREE separate channels. They share nothing: each has its own
detection, its own timers and its own message, so one channel can never delay, replace or suppress
another.

    ml      The trained per-AC model (Isolation Forest + distance-to-normal) says the reading is not how
            THIS AC normally behaves. Works even when every reading is inside the safety limits.
            Never touched by the Safety Limit Alerts switch. With no usable model it stays silent:
            a threshold is never presented as an ML result.
    safety  Predefined safety limits (the per-AC limits on the Manage ACs page). Can be switched off with
            "Safety Limit Alerts". Current / voltage / power over the limit alert at once; every other limit
            must still be a problem 3 minutes later. High humidity is NOT a limit any more (low humidity is).
    sensor  A sensor that did not answer (offline). It is a device problem, not a threshold and not an ML
            result, so neither switch affects it.

This file has no web or database code, so the whole decision can be tested with real model results.
"""
import threading

from services import ml_explain

# --- ALERT TIMING (per AC, per channel) ------------------------------------------------------------
# URGENT (INSTANT) problems alert at once, the moment the condition is seen:
#   * current / voltage / power over the limit (the "instant" limits on Manage ACs),
#   * vibration or a compressor temperature over the limit WHILE the model also says the AC is abnormal,
#   * (ML channel) the model finds a CRITICAL departure from normal, or current / voltage / power far above this AC's normal.
# NOT-URGENT problems (the model sees odd readings, a sensor is offline, any other limit...) must still be
# there SOFT_DELAY_S after they started before they alert.
# After an alert a channel waits CRITICAL_REPEAT_S before sending the SAME KIND again. The pause is per channel
# AND per kind: a normal alert never makes an instant one wait, and a safety alert never makes an ML alert wait.
CRITICAL_REPEAT_S = 120
SOFT_DELAY_S = 180
STILL_WINDOW_S = 60       # "still a problem" = at least half of the last minute's readings had one
# How many readings in a row must show an INSTANT problem before it goes out. The instant safety limits (current / voltage / power)
# alert on the very first reading. A severe ML finding needs two in a row (about 10 s): one reading can be a sensor glitch or a
# compressor start, and real AC data has a few such outliers.
INSTANT_CONFIRM_READINGS = {"ml": 2}
INSTANT_MIN_GAP_S = 30    # a NEW instant problem (the old one had gone away) may alert again after this long, not only after CRITICAL_REPEAT_S

CHANNELS = ("ml", "safety", "sensor")
CHANNEL_LABEL = {"ml": "ML", "safety": "Safety", "sensor": "Sensor"}

REQUIRED_SENSORS = ["dht_temp", "dht_humidity", "pzem_voltage", "pzem_current", "ds18b20_temp1", "ds18b20_temp2", "vibration"]

# rule-engine findings that mean "a sensor is offline" (they belong to the sensor channel, not the safety one)
OFFLINE_RULE_IDS = {"dht_offline", "pzem_offline", "temp1_offline", "temp2_offline", "vibration_offline"}


def limit_sets(L):
    """The limits that make a safety alert URGENT (instant): electrical safety, whatever else is going on."""
    return [
        ("pzem_current", L["current"]["max"], "Current", "A", False),
        ("pzem_voltage", L["voltage"]["max"], "Voltage", "V", False),
        ("pzem_power", L["power"]["max"], "Power", "W", False),
    ]


def limit_breaches(payload_dict, limits):
    """Readable sentences for every limit that is broken in the raw numbers."""
    out = []
    for key, limit, name, unit, inclusive in limits:
        try:
            v = float(payload_dict.get(key))
        except (TypeError, ValueError):
            continue
        if v and (v >= limit if inclusive else v > limit):
            out.append((f"{name} is {v:g} {unit}".rstrip()) + (f", limit is {limit:g} {unit}".rstrip()))
    return out


def escalation_limits(L):
    """Compressor-protection limits: not instant on their own (a vibration or temperature blip is common), but INSTANT
    when the trained model also says the AC is abnormal right now."""
    return [
        ("vibration", L["vibration"]["max"], "Vibration", "", True),
        ("ds18b20_temp1", L["outlet_temp"]["max"], "Outlet compressor temperature", "C", False),
        ("ds18b20_temp2", L["inlet_temp"]["max"], "Inlet compressor temperature", "C", False),
    ]


def offline_sensors(payload_dict):
    return [f"Sensor offline: {key}" for key in REQUIRED_SENSORS if payload_dict.get(key) is None]


def ml_state(ml, unit_has_model=True):
    """'normal' / 'anomaly' / 'no_model' from a detector result (None = no model for this AC)."""
    if ml is None:
        return "no_model"
    return "anomaly" if ml.get("is_anomaly") else "normal"


def evaluate_channels(payload, unit_limits, ml, safety_on, rule_result, build_problems, unit_label=None):
    """Judge ONE reading on the three channels.

    payload        the raw reading (dict)
    unit_limits    this AC's safety limits
    ml             result of the trained model for this AC, or None when it has no (usable) model
    safety_on      the "Safety Limit Alerts" switch
    rule_result    analyze_with_rules(payload, unit_limits)
    build_problems function(critical_reasons, findings, ml) -> list of plain problem lines

    Returns {channel: {"crit": bool, "soft": bool, "snapshot": dict | None}} for every channel.
    The ML channel looks ONLY at `ml`. The safety channel never decides WHETHER to alert from `ml`; the model only makes
    a compressor-limit alert INSTANT (it can speed a safety alert up, never hold one back).
    """
    findings = list(rule_result.get("findings") or [])
    limit_findings = [f for f in findings if f.get("sensor_key") not in OFFLINE_RULE_IDS]
    sensor_findings = [f for f in findings if f.get("sensor_key") in OFFLINE_RULE_IDS]

    out = {}

    # ---- ML channel: the trained model only ----
    ml_flag = bool(ml is not None and ml.get("is_anomaly"))
    assessment = ml_explain.assess(ml, payload, unit_label) if ml_flag else None
    out["ml"] = {
        # INSTANT for a severe finding (ml_explain.instant_reason); every other ML finding must still be abnormal after SOFT_DELAY_S
        "crit": bool(assessment and assessment.get("instant")),
        "soft": ml_flag,
        "snapshot": ({"payload": dict(payload), "findings": [], "ml": ml, "limits": unit_limits, "assessment": assessment,
                      "problems": assessment["lines"] if assessment else build_problems([], [], ml), "source": "ml"} if ml_flag else None),
    }

    # ---- Safety channel: predefined limits only, and only while the switch is on ----
    if safety_on:
        crucial = limit_breaches(payload, limit_sets(unit_limits))
        escalated = limit_breaches(payload, escalation_limits(unit_limits)) if ml_flag else []
        crit, soft = bool(crucial or escalated), bool(limit_findings)
        snap = ({"payload": dict(payload), "findings": limit_findings, "ml": None, "limits": unit_limits,
                 "problems": build_problems(crucial + escalated, limit_findings, None), "source": "safety"} if (crit or soft) else None)
    else:
        crit, soft, snap = False, False, None
    out["safety"] = {"crit": crit, "soft": soft, "snapshot": snap}

    # ---- Sensor channel: something is offline ----
    offline = offline_sensors(payload)
    out["sensor"] = {
        "crit": False,
        "soft": bool(offline),
        "snapshot": ({"payload": dict(payload), "findings": sensor_findings, "ml": None, "limits": unit_limits,
                      "problems": build_problems(offline, sensor_findings, None), "source": "sensor"} if offline else None),
    }
    return out


class AlertScheduler:
    """Per AC AND per channel timers: when does a problem become an alert?

    Instant (critical) and normal (soft) alerts keep SEPARATE repeat pauses: an instant alert only waits for the
    last INSTANT alert of its channel, never for a normal one (that was the bug that held instant alerts back).
    A normal alert still waits after any alert of its channel, so it does not follow one straight away.
    """

    def __init__(self, repeat_s=None, soft_delay_s=None, window_s=None):
        self.repeat_s = CRITICAL_REPEAT_S if repeat_s is None else repeat_s
        self.soft_delay_s = SOFT_DELAY_S if soft_delay_s is None else soft_delay_s
        self.window_s = STILL_WINDOW_S if window_s is None else window_s
        self._state = {}
        self._lock = threading.Lock()

    def _get(self, key):
        st = self._state.get(key)
        if st is None:
            st = self._state[key] = {"last_crit": float("-inf"), "last_soft": float("-inf"),
                                     "soft_since": None, "recent": [], "snapshot": None, "new_episode": True, "crit_streak": 0}
        return st

    def snapshot(self, unit, channel):
        with self._lock:
            return dict(self._get((unit, channel)).get("snapshot") or {})

    def reset(self, unit=None):
        with self._lock:
            if unit is None:
                self._state.clear()
            else:
                for key in [k for k in self._state if k[0] == unit]:
                    del self._state[key]

    def decide(self, unit, channel, now, crit_now, soft_now, snapshot=None):
        """Once per reading and channel. Returns "critical", "soft" or None.

        `snapshot` is the latest reading of this channel that had a problem, kept so the message describes
        what is wrong at the moment it is sent.
        """
        with self._lock:
            st = self._get((unit, channel))
            need = INSTANT_CONFIRM_READINGS.get(channel, 1)
            st["crit_streak"] = st["crit_streak"] + 1 if crit_now else 0
            if crit_now and st["crit_streak"] < need:
                crit_now, soft_now = False, True          # not confirmed yet: counts as a normal problem for now
            present = bool(crit_now or soft_now)
            if snapshot is not None:
                st["snapshot"] = snapshot
            st["recent"] = [(t, f) for t, f in st["recent"] if now - t <= self.window_s] + [(now, present)]
            still_there = sum(1 for _, f in st["recent"] if f) * 2 >= len(st["recent"])
            if present and st["soft_since"] is None:
                st["soft_since"] = now                 # a problem episode starts
            if not present and not still_there:
                st["soft_since"] = None                # it went away: forget it
                st["new_episode"] = True               # the next instant problem is a NEW one: it must not wait the full repeat pause
            crit_cooled = (now - st["last_crit"] >= self.repeat_s
                           or (st["new_episode"] and now - st["last_crit"] >= INSTANT_MIN_GAP_S))
            soft_cooled = now - max(st["last_crit"], st["last_soft"]) >= self.repeat_s

            action = None
            if crit_now and crit_cooled:
                action = "critical"                    # instant: at once, whatever normal alerts were sent before
                st["last_crit"] = now
                st["new_episode"] = False
            elif (st["soft_since"] is not None and now - st["soft_since"] >= self.soft_delay_s
                  and still_there and soft_cooled and not crit_now):
                action = "soft"                        # not instant: still there after the delay
                st["last_soft"] = now
            if action:
                st["soft_since"] = now                 # next normal one after the delay again
            return action
