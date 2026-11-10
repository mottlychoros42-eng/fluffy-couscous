import time
import json
import re
from services.ai_predictor import ask
from services.traccar_sms import send_sms
from services.alerts_api import create_alert, AlertIn, update_alert_diagnoses
from services.rule_engine import analyze_with_rules, _fmt_like
from services.alert_channels import OFFLINE_RULE_IDS, CHANNEL_LABEL
from services import ml_explain
from services.limits import merged
from services import action_guide

# the sensors shown on the alert page
READING_KEYS = ["pzem_power", "pzem_voltage", "pzem_current", "dht_temp", "dht_humidity",
                "ds18b20_temp1", "ds18b20_temp2", "vibration"]

def format_for_sms(api_response_string):
    """
    Parses the JSON AI response and formats it for an SMS message.
    Limits output to High/Medium severity to save characters.
    """
    try:
        # Parse the JSON string into a Python dictionary
        data = json.loads(api_response_string)
        diagnoses = data.get("diagnoses", [])
    except json.JSONDecodeError:
        return "⚠️ Alert: Diagnostics completed, but data formatting failed."

    # Filter for High and Medium severity issues
    urgent_issues = [d for d in diagnoses if d.get("severity") in ["High", "Medium"]]

    # If everything is Low severity/Normal
    if not urgent_issues:
        return "✅ AC Status: Normal. No urgent issues detected."

    # Build the SMS string
    sms_message = "🚨 AC Diagnostic Alert 🚨\n"
    
    # Limit to the top 2 issues to prevent the SMS from getting too long
    for issue in urgent_issues[:2]:
        title = issue.get("issue", "Unknown")
        severity = issue.get("severity", "Unknown").upper()
        
        # Take just the first sentence of the recommended action to save space
        action_full = issue.get("recommended_action", "Inspect unit.")
        parts = [p.strip() for p in action_full.split(". ") if p.strip()]
        if parts:
            if parts[0].isdigit() and len(parts) > 1:
                action_short = f"{parts[0]}. {parts[1]}."
            else:
                action_short = f"{parts[0]}."
        else:
            action_short = "Inspect unit."
        
        sms_message += f"\n[{severity}] {title}\nAction: {action_short}\n"

    sms_message += "\nCheck full system report for details."
    
    return sms_message

def ac_status_to_str(status):
    if status == 0:
        return "Off"
    elif status == 1:
        return "Fan"
    elif status == 2:
        return "Low Cool"
    elif status == 3:
        return "Medium Cool"
    elif status == 4:
        return "High Cool"
    else:
        return "Unknown"

# ==========================================
# 1. THE COMBINED PROMPT TEMPLATE
# Combines the AI's persona, knowledge base, formatting rules,
# and the specific telemetry data into a single string.
# ==========================================
_PROMPT_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")


def _apply_limits(text, L):
    """The AI prompt describes the default AC; swap its numbers for this AC's own limits."""
    pairs = [(1650, L["power"]["min"]), (1860, L["power"]["max"]), (7.6, L["current"]["min"]), (8.8, L["current"]["max"]),
             (225, L["voltage"]["min"]), (241, L["voltage"]["max"]), (7, L["output_temp"]["min"]), (25, L["output_temp"]["max"]),
             (90, L["vibration"]["max"]), (50, L["outlet_temp"]["min"]), (61, L["outlet_temp"]["max"]),
             (8, L["inlet_temp"]["min"]), (17, L["inlet_temp"]["max"])]

    def sub(m):
        tok = m.group(1)
        for old, new in pairs:
            if float(tok) == float(old):
                return tok if float(new) == float(old) else _fmt_like(tok, new)
        return tok
    return _PROMPT_NUM.sub(sub, text)


def generate_combined_prompt(telemetry_data, ml_note=None):
    # Note: Double curly braces {{ }} are used below to escape 
    # the JSON brackets so the Python f-string doesn't break.
    return f"""You are an expert HVAC diagnostic AI. Your task is to analyze IoT sensor telemetry from a Window Type Air Conditioner and perform Proactive Fault Detection. You are not limited to identifying a single issue; you should detect multiple possible failures currently affecting the AC, as well as predict possible future failures based on the telemetry trends.

SYSTEM SPECIFICATIONS:
- Capacity: 1.5 HP
- Type: Manual Window Type (Non-Inverter)
- Refrigerant: R410A
- Expected Watts: 1650W - 1860W
- Expected Current: 7.6A - 8.8A
- Voltage/Frequency: 225-241V / 60Hz / 1Ph
- Return Air (AC Output Temp/Humidity): Measures room/supply air. Expected normal range is 7-25 °C. Humidity is monitored for information only: do NOT report high humidity as an issue (cooled air is normally humid); report humidity only if it is abnormally LOW.
- Compressor Vibration: Mounted on the compressor. Expected normal range is less than 90 Hz. Vibration >= 90 Hz indicates mechanical wear, loose mounts, or severe compressor overload.
- Discharge Line Temp (Outlet Compressor Temp): Leaving the compressor. Expected normal range is 50-61 °C. Abnormally high temps (> 61 °C) suggest a dirty condenser coil or low refrigerant.
- Suction Line Temp (Inlet Compressor Temp): Leaving the indoor coil. Expected normal range is 8-17 °C. Temperatures < 8 °C suggest evaporator freezing risk.
- Power Monitor: Main input. Power > 1860W suggests an overworked compressor or capacitor failing. Power < 1650W (but > 0) suggests the fan is running but the compressor failed to start. Current > 8.8A suggests overloading; current < 7.6A suggests underload or failed start.
- Refrigerant Leak Signature: Triggered when there is a combination of: Low Current (< 7.6A), Low Power (< 1700W), High Output Air Temp (> 25°C), and High Compressor Outlet Temp (> 70°C). Diagnosing a Refrigerant Leak maps to "Maintenance" (Need Maintenance) action.

INSTRUCTIONS:
Analyze the provided telemetry data. Provide a structured response detailing all current and potential future issues.
Format your response as a JSON object containing a "diagnoses" array. Each object in the array must have the following keys:
{{
  "diagnoses": [
    {{
      "issue": "Clear explanation of the suspected or predicted issue",
      "status": "Current or Predicted",
      "confidence_score": "Percentage (0-100%) indicating how certain the AI is about this diagnosis",
      "root_cause": "The specific sensor readings that lead to this conclusion",
      "severity": "Low, Medium, High, or Critical",
      "recommended_action": "Actionable steps for the user or technician"
    }}
  ]
}}

--------------------------------------------------

An anomaly trigger has been activated. Please analyze the following real-time sensor telemetry:
{ml_note or ""}
{json.dumps(telemetry_data, indent=4)}

Based on the AC specifications and the sensor context provided, what are the current and potential future issues?
"""


def _scope_rule_result(rule_result, source):
    """Keep only what belongs to this alert's source, so the alert never shows another channel's findings.

    ml      -> no threshold findings at all (an ML alert is the model's opinion only)
    safety  -> threshold findings, without the sensor-offline ones
    sensor  -> only the sensor-offline findings
    """
    findings = list(rule_result.get("findings") or [])
    if source == "ml":
        keep_f, keep_s = [], []
    elif source == "sensor":
        keep_f = [f for f in findings if f.get("sensor_key") in OFFLINE_RULE_IDS]
        keep_s = [c for c in rule_result.get("computation_steps", []) if c.get("sensor_key") in OFFLINE_RULE_IDS]
    else:
        keep_f = [f for f in findings if f.get("sensor_key") not in OFFLINE_RULE_IDS]
        keep_s = [c for c in rule_result.get("computation_steps", []) if c.get("sensor_key") not in OFFLINE_RULE_IDS]
    rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    top = max(keep_f, key=lambda f: rank.get(f.get("severity"), 0), default=None)
    return {
        "findings": keep_f,
        "overall_severity": (top or {}).get("severity") or "Normal",
        "sensors_checked": len(keep_s),
        "sensors_triggered": len(keep_f),
        "compressor_off": rule_result.get("compressor_off", False),
        "computation_steps": keep_s,
    }


def _ml_rule_result(assessment, base):
    """The 'Problems found' block of an ML alert: the odd sensors found by the model, in the same shape as rule findings."""
    steps = [{
        "sensor": f["sensor"], "sensor_key": f["sensor_key"], "value": f["measured_value"], "normal_range": f["normal_range"],
        "result": "TRIGGERED", "expression": f["computed_expression"], "severity": f["severity"], "issue": f["issue"],
    } for f in assessment["findings"]]
    return {
        "findings": assessment["findings"],
        "overall_severity": assessment["severity"],
        "sensors_checked": len(steps),
        "sensors_triggered": len(steps),
        "compressor_off": base.get("compressor_off", False),
        "computation_steps": steps,
    }

# ==========================================
# 2. YOUR REFACTORED FUNCTION
# ==========================================
def process_anomaly(anomaly_data):
    """
    Process the detected anomaly using a two-layer analysis:

    LAYER 1 — Rule Engine (deterministic, physics-based)
      Runs FIRST and ALWAYS. Checks each sensor against hard-coded
      thresholds derived from the AC unit specifications. Produces
      a structured list of findings with severity, root cause,
      and a step-by-step computation audit trail.

    LAYER 2 — AI / LLM (contextual, probabilistic)
      Runs SECOND if the rule engine confirms something is anomalous
      OR if the Isolation Forest flagged it. Provides deeper context
      and predicted future failures. May be unavailable (API down).

    Both results are stored together in the alert's `diagnoses` field
    as a combined JSON: { "rule_diagnoses": {...}, "ai_diagnoses": {...} }
    """
    ml_info = anomaly_data.pop("_ml", None)   # what the per-AC model found odd (may be None)
    L = merged(anomaly_data.pop("_limits", None))   # this AC's own limits (defaults if none)
    unit_title = anomaly_data.pop("_title", None)   # AC name, used in the SMS
    is_critical = bool(anomaly_data.pop("_critical", False))
    source = anomaly_data.pop("_source", None) or "safety"   # "ml", "safety" or "sensor": where this alert comes from
    received_at = anomaly_data.pop("_received_at", None)     # when the reading arrived (only for the latency log)
    if source != "ml":
        ml_info = None        # a safety / sensor alert is not an ML result and must not look like one
    print(f"Processing raw anomaly data ({source} alert)...")

    # ── LAYER 1: Rule Engine ───────────────────────────────────
    # Pass the raw numeric data directly to the rule engine
    # (it uses the same keys as the telemetry payload)
    raw_sensor_data = dict(anomaly_data)  # preserve original for rule engine
    rule_result = _scope_rule_result(analyze_with_rules(raw_sensor_data, L), source)

    # An ML alert explains what the model found, in the same detail as a rule-based alert: the odd sensors with their
    # real values and this AC's usual range, why the model calls it abnormal, severity, risks and what to do.
    assessment = None
    if source == "ml" and ml_info:
        assessment = ml_explain.assess(ml_info, raw_sensor_data, unit_title)
        if assessment:
            rule_result = _ml_rule_result(assessment, rule_result)

    print("\n--- RULE ENGINE RESULT ---")
    print(json.dumps(rule_result, indent=2))
    print("--------------------------\n")

    # ── Build human-readable keys for AI prompt ────────────────
    clean_data = {
        "Return Air Temperature (C)": anomaly_data.pop("dht_temp", None),
        "Return Air Humidity (%)": anomaly_data.pop("dht_humidity", None),
        "Compressor Vibration (Hz)": anomaly_data.pop("vibration", None),
        "Compressor Discharge Line Temp (C)": anomaly_data.pop("ds18b20_temp1", None),
        "Compressor Suction Line Temp (C)": anomaly_data.pop("ds18b20_temp2", None),
        "Voltage (V)": anomaly_data.pop("pzem_voltage", None),
        "Current (A)": anomaly_data.pop("pzem_current", None),
        "Power (W)": anomaly_data.pop("pzem_power", None),
        "Energy (kWh)": (anomaly_data.pop("pzem_energy", None) / 1000.0) if anomaly_data.get("pzem_energy") is not None else None,
        "Frequency (Hz)": anomaly_data.pop("pzem_frequency", None),
        "Power Factor": anomaly_data.pop("pzem_power_factor", None),
        "AC Status": ac_status_to_str(anomaly_data.pop("ac_status", None)),
        "AC Thermostat": anomaly_data.pop("ac_thermostat", None)
    }
    clean_data = {k: v for k, v in clean_data.items() if v is not None}

    # Generate the combined AI prompt
    ml_note = None
    if source == "ml" and ml_info:
        odd = "; ".join(f"{o['name']} {o['value_text']} (usually {o['usual_text']})" for o in (assessment or {}).get("odd", [])) \
            or ", ".join(action_guide.ml_names(ml_info)) or "no single sensor stands out"
        ml_note = (f"This alert comes from the trained machine-learning model of this AC, not from a safety limit. "
                   f"It judged the readings abnormal compared with how THIS AC normally behaves (odd readings: {odd}). "
                   f"Explain what that pattern could mean.\n")
    combined_prompt_text = _apply_limits(generate_combined_prompt(clean_data, ml_note), L)

    print("\n--- PROMPT READY FOR API ---")
    print(combined_prompt_text)
    print("----------------------------\n")

    # ── Build the alert and a short, plain SMS (no links: carriers block them) ─
    ai_pending = {"pending": True, "note": "The AI opinion is being prepared and will appear here. Refresh in a minute."}
    diagnoses = {
        "source":         source,                       # "ml" | "safety" | "sensor"
        "source_label":   CHANNEL_LABEL.get(source, source),
        "rule_diagnoses": rule_result,
        "ai_diagnoses":   ai_pending,
        "readings":       {k: raw_sensor_data.get(k) for k in READING_KEYS},
        "limits":         L,
        "action_plan":    action_guide.build_plan(rule_result["findings"], ml_info, assessment),
        "ml":             ml_info,
        "ml_assessment":  assessment,                   # severity, condition, why, risks, actions (ML alerts only)
    }
    sms_message = action_guide.sms_text(unit_title, is_critical, rule_result["findings"], ml_info, source, assessment)

    print("\n--- FORMATTED SMS ---")
    print(sms_message)
    print("---------------------\n")

    # ── 1) SAVE the alert and 2) SEND the SMS first: nothing slow (the AI) comes before them ─
    # The SMS must never depend on the database: if saving fails the SMS still goes out (just without a Report ID).
    alert = None
    try:
        alert = create_alert(AlertIn(summary=sms_message, diagnoses=json.dumps(diagnoses)))
    except Exception as save_err:
        print(f"\n[ERROR] could not save the alert (the SMS is still sent): {save_err}\n", flush=True)
    try:
        # Note: Do not append the HTTP URL because carriers block SMS messages containing links.
        send_sms(sms_message + (("\nReport ID: " + str(alert.id)) if alert else ""))
        if received_at:
            print(f"[LATENCY] {unit_title} {source}: SMS handed to the gateway {time.time() - received_at:.2f}s after the reading arrived", flush=True)
    except Exception as sms_err:
        print(f"\n--- SMS SEND FAILED (alert still saved) ---\n{sms_err}\n-------------------------------------------\n")
    if alert is None:
        return

    # ── 3) LAYER 2: AI diagnosis, last. It can take a minute; the alert and the SMS are already out ─
    ai_diagnoses_payload = _ask_ai(combined_prompt_text)
    diagnoses["ai_diagnoses"] = ai_diagnoses_payload
    try:
        update_alert_diagnoses(alert.id, json.dumps(diagnoses))
    except Exception as e:
        print(f"\n[ERROR] could not attach the AI opinion to alert {alert.id}: {e}\n", flush=True)


def _ask_ai(prompt_text):
    """The AI opinion (parsed JSON), or {"error": ...} when it is unavailable. Never raises."""
    try:
        api_response = ask(prompt_text)
        print("\n--- AI RESPONSE ---")
        print(api_response)
        print("-------------------\n")
        try:
            cleaned_response = api_response.strip() if isinstance(api_response, str) else ""
            if cleaned_response.startswith("```"):          # the LLM wrapped its answer in a markdown code block
                lines = cleaned_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned_response = "\n".join(lines).strip()
            start_idx, end_idx = cleaned_response.find("{"), cleaned_response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_response = cleaned_response[start_idx: end_idx + 1]
            return json.loads(cleaned_response) if cleaned_response else api_response
        except json.JSONDecodeError:
            return {"raw": api_response}
    except Exception as e:
        err_text = str(e)
        print(f"\n--- AI PREDICTION ERROR ---\n{err_text}\n---------------------------\n")
        return {"error": err_text, "note": "AI layer unavailable; the checks on the left are still valid."}
