"""
Probable causes from the COMBINATION of odd sensors (v3.5).

The model says which sensors are off and in which direction (above / below this AC's usual value). One sensor alone rarely
says why: a high current has a different cause when the supply voltage is also high than when the compressor is hot, and
"clean the air filter" is the right advice for a freezing coil, not for an over-voltage. So instead of canned advice per
sensor, this module looks at the pattern, names 1 to 3 probable causes, shows the readings that support each one, and lists
every action only ONCE across the whole plan.

Pure Python, no web or database code. Input: the `odd` entries made by ml_explain._sensor_entry().
"""

A, B = "above", "below"
V, I, P, PF, HZ = "pzem_voltage", "pzem_current", "pzem_power", "pzem_power_factor", "pzem_frequency"
AIR, HUM, OUT, INL, VIB = "dht_temp", "dht_humidity", "ds18b20_temp1", "ds18b20_temp2", "vibration"

SEV_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

# need:    every group must have at least one of its (sensor, direction) pairs present
# support: extra pairs that raise the confidence when present
# exclude: if any of these is present the pattern does not apply (another pattern explains it better)
# urgent:  a pattern that can damage the AC quickly: shown as "stop and check now" when the finding is severe
PATTERNS = [
    dict(id="overvoltage", priority=1, urgent=True, short="supply voltage too high",
         title="The electricity supply voltage is too high",
         need=[[(V, A)]], support=[(I, A), (P, A)],
         cause="The building's supply is above what this AC normally gets. At high voltage the AC draws more current and power, "
               "so those readings rise with it instead of pointing to a fault inside the AC.",
         risk="Over-voltage can burn out the compressor and the control board.",
         actions=["Measure the voltage at the outlet or breaker with a multimeter to confirm it.",
                  "Ask the building's electrician to check the incoming supply and the voltage regulator (AVR).",
                  "Switch the AC off at the breaker if the voltage stays high. Do not keep running it."],
         check=["Main supply and voltage regulator (AVR)."],
         call="the voltage stays above the AC's rated range for more than a few minutes."),
    dict(id="undervoltage", priority=1, urgent=True, short="supply voltage too low",
         title="The electricity supply voltage is too low",
         need=[[(V, B)]], support=[(I, A), (OUT, A)],
         cause="The supply is below what this AC normally gets, so the compressor draws extra current to make up for it and heats up.",
         risk="Low voltage overheats the motors and can stop the compressor from starting.",
         actions=["Switch off other large appliances on the same line (heaters, pumps, other ACs) and see if the voltage recovers.",
                  "Check the breaker and the wiring for loose or warm connections: they cause a voltage drop.",
                  "Switch the AC off if the voltage stays low."],
         check=["Wiring and breaker terminals.", "Voltage regulator (AVR)."],
         call="the voltage stays low."),
    dict(id="compressor_heat", priority=2, urgent=True, short="compressor running too hot",
         title="The compressor is running too hot",
         need=[[(OUT, A)]], support=[(VIB, A), (INL, A), (I, A), (P, A), (AIR, A)],
         cause="The compressor's outlet pipe is hotter than this AC normally gets. Usual causes: the outdoor coil cannot shed heat "
               "(dirty or blocked), the outdoor fan is not moving enough air, there is too little refrigerant, or the compressor is "
               "overloaded (for example by a high supply voltage).",
         risk="An overheating compressor trips its protector or can seize.",
         actions=["Switch the AC off for about 15 minutes so the compressor cools down, and do not restart it again and again.",
                  "Check the outdoor unit: is the fan spinning, and is the coil clogged with dirt or leaves or blocked by something?",
                  "With the power off, rinse the outdoor coil gently with water and clear the space around the unit."],
         check=["Outdoor coil and fan motor.", "Refrigerant level."],
         call="it overheats again after the outdoor unit has been cleaned."),
    dict(id="compressor_not_running", priority=2, urgent=False, short="compressor not running",
         title="The compressor does not seem to be running",
         need=[[(I, B)], [(P, B)]], support=[(VIB, B), (AIR, A), (OUT, B), (PF, B)], exclude=[(I, A), (P, A), (OUT, A)],
         where=lambda o: o[I]["value"] < 0.5 * o[I]["usual_mid"],       # a mild drop is NOT "not running"
         cause="Current, power and vibration are all very low while the air coming out is warm, so only the fans are working "
               "(or nothing is) and the compressor is not doing its job.",
         risk="No cooling, and a compressor that keeps trying to start can burn out.",
         actions=["Check the thermostat: set it clearly below the room temperature, in cool mode.",
                  "Listen at the outdoor unit: the compressor should hum and the outdoor fan should spin.",
                  "Check the breaker and the outdoor unit's power. A tripped overload protector or a failed start capacitor stops the compressor."],
         check=["Start capacitor and overload protector."],
         call="the compressor still does not start after resetting the breaker."),
    dict(id="overload", priority=3, urgent=True, short="AC drawing too much electricity",
         title="The AC is drawing more electricity than normal",
         need=[[(I, A)], [(P, A)]], support=[(OUT, A), (VIB, A)], exclude=[(V, A)],
         cause="Current and power are both up but the supply voltage is not, so the AC itself is working harder: a struggling motor, "
               "a weak run capacitor, or an outdoor coil that cannot release heat.",
         risk="Overloaded motors overheat and trip the breaker.",
         actions=["Check that the indoor and outdoor fans spin freely and the outdoor coil is not blocked.",
                  "Listen for humming, clicking or a struggling start from the compressor, and have the run capacitor tested if you hear it.",
                  "Feel the breaker and wiring for warmth, and switch the AC off if you smell burning."],
         check=["Run capacitor.", "Fan and compressor motors."],
         call="the current stays high after the coil is clean."),
    dict(id="low_refrigerant", priority=3, urgent=False, short="possible low refrigerant",
         title="Cooling capacity has dropped: probably low refrigerant",
         need=[[(I, B), (P, B)], [(AIR, A), (INL, A)]], support=[(AIR, A), (INL, A), (OUT, A), (HUM, A)],
         cause="The AC runs but is not removing heat: the electrical draw is lower than usual while the air coming out and the inlet "
               "pipe are warmer. That pattern usually means too little refrigerant, often a slow leak.",
         risk="Running on low refrigerant ices the coil and wears out the compressor.",
         actions=["Look for oily stains at the pipe joints and flare nuts: they show a leak.",
                  "Feel the thicker pipe at the outdoor unit: it should be cold. If it is warm, cooling is weak.",
                  "Have a technician measure the refrigerant pressure and find the leak. Topping up without fixing the leak only delays the problem."],
         check=["Refrigerant charge and pipe joints."],
         call="the air stays warm or ice keeps forming on the pipes."),
    dict(id="airflow_icing", priority=3, urgent=False, short="weak airflow, coil icing risk",
         title="Not enough air is flowing over the indoor coil (icing risk)",
         need=[[(AIR, B), (INL, B)]], support=[(AIR, B), (INL, B), (I, B), (P, B)], exclude=[(AIR, A)],
         cause="The air coming out and the inlet pipe are colder than usual: the coil is getting too cold because too little room air "
               "passes over it. A dirty filter, a slow blower or blocked vents are the usual reasons.",
         risk="The coil can freeze over and liquid refrigerant can damage the compressor.",
         actions=["Clean or replace the air filter.",
                  "Check that the indoor fan spins and that no vent or grille is blocked.",
                  "If you see ice, switch the AC off and let it thaw for an hour or two before restarting."],
         check=["Indoor fan and evaporator coil."],
         call="ice comes back after the filter is clean."),
    dict(id="vibration", priority=4, urgent=False, short="compressor vibrating",
         title="The compressor is vibrating more than usual",
         need=[[(VIB, A)]], exclude=[(OUT, A)],
         cause="More vibration with no other sign of trouble points to loose mounts, a worn bearing or an unbalanced fan blade.",
         risk="Vibration can crack the refrigerant pipes and wear out bearings quickly.",
         actions=["Listen for rattling, grinding or humming and note where it comes from.",
                  "Check the compressor mounting bolts and rubber feet, and the fan blades for dirt or damage.",
                  "Make sure no pipe or panel is touching the cabinet."],
         check=["Mounts, bearings and fan blades."],
         call="the noise or vibration continues."),
    dict(id="weak_cooling", priority=4, urgent=False, short="air warmer than usual",
         title="The air is warmer than usual although the AC runs normally",
         need=[[(AIR, A)]], support=[(HUM, A), (INL, A)], exclude=[(I, A), (I, B), (P, A), (P, B)],
         cause="Current and power are normal, so the AC is running as usual. The room may have an unusual heat load, or heat "
               "exchange is reduced a little.",
         risk="Poor comfort, and the AC runs longer than it should.",
         actions=["Check for open doors or windows and strong heat sources in the room (sun, ovens, many people).",
                  "Check that the vents are open and the air flow feels normal.",
                  "Check that the outdoor unit is not blocked or sitting in hot exhaust air."],
         check=["Room heat load.", "Outdoor unit airflow."],
         call="the air stays warm for more than a day."),
    dict(id="power_factor", priority=4, urgent=False, short="motor running inefficiently",
         title="The motor is not running efficiently (weak run capacitor)",
         need=[[(PF, B)]], support=[(I, A)], exclude=[(I, B)],
         cause="A low power factor means the motor is wasting electricity, usually because the run capacitor is weak or a motor is struggling.",
         risk="A weak capacitor heats the motor and shortens its life.",
         actions=["Listen for humming or a slow start from the compressor or fan.",
                  "Have the run capacitor tested and replaced if it is weak."],
         check=["Run capacitor."], call="the AC hums or struggles to start."),
    dict(id="frequency", priority=5, urgent=False, short="unstable supply frequency",
         title="The electricity frequency is unstable",
         need=[[(HZ, A), (HZ, B)]],
         cause="The supply frequency is off its usual value, which happens on a generator or an unstable grid.",
         risk="An unstable frequency stresses the motors.",
         actions=["Check whether the building is running on a generator or backup power.",
                  "Ask the electrician to check the generator or the supply."],
         check=["Power source."], call="the frequency keeps drifting."),
    dict(id="humidity_high", priority=5, urgent=False, short="air more humid than usual",
         title="The air coming out is more humid than usual",
         need=[[(HUM, A)]], exclude=[(AIR, A)],
         cause="Wetter air coming out is often a blocked water drain or low airflow. Humidity is shown for information, not as a limit.",
         risk="Water can leak inside and mould can grow.",
         actions=["Check that the water drain is not blocked and the drain pan is not full.",
                  "Check that the air flows normally through the unit."],
         check=["Drain line."], call="water drips inside or the air feels clammy for days."),
    dict(id="humidity_low", priority=5, urgent=False, short="air drier than usual",
         title="The air coming out is drier than usual",
         need=[[(HUM, B)]],
         cause="Drier air than normal can mean the humidity sensor is misreading or the air flow changed.",
         risk="Mostly comfort, unless the sensor is faulty.",
         actions=["Check the sensor and the airflow through the unit."],
         check=["DHT22 sensor."], call="it stays low for days."),
    dict(id="sensor_fault", priority=6, urgent=False, short="possible faulty sensor", single_of=[AIR, HUM, OUT, INL],
         title="This may be a faulty sensor (only this reading is off)",
         need=[],
         cause="Everything else looks normal, so a loose, wet or faulty sensor is possible.",
         risk="A wrong reading can hide a real problem or raise a false alarm.",
         actions=["Check the sensor wiring and that the probe is firmly in place.",
                  "Compare with a handheld thermometer or meter."],
         check=["Sensor and its cable."], call="the reading stays wrong after re-seating the sensor."),
]


def _evidence(covered, odd_by_key):
    bits = []
    for k in covered:
        o = odd_by_key.get(k)
        if o:
            bits.append(f"{o['short']} {o['value_text']} (usually {o['usual_text']})")
    return bits


def diagnose(odd):
    """Return (hypotheses, unexplained_odd_sensors). Each hypothesis is a dict ready to become a plan step."""
    if not odd:
        return [], []
    dirs = {o["key"]: o["direction"] for o in odd}
    odd_by_key = {o["key"]: o for o in odd}
    cands = []
    for p in PATTERNS:
        if any(dirs.get(k) == d for k, d in p.get("exclude", [])):
            continue
        if p.get("single_of"):
            if len(odd) == 1 and odd[0]["key"] in p["single_of"]:
                cands.append((p, [(odd[0]["key"], odd[0]["direction"])], [], 0.5))
            continue
        matched = []
        for alts in p["need"]:
            hit = next(((k, d) for k, d in alts if dirs.get(k) == d), None)
            if hit is None:
                matched = None
                break
            matched.append(hit)
        if matched is None:
            continue
        if p.get("where"):
            try:
                if not p["where"](odd_by_key):
                    continue
            except Exception:
                continue
        sup = [(k, d) for k, d in p.get("support", []) if dirs.get(k) == d]
        total = len(p["need"]) + len(p.get("support", []))
        cands.append((p, matched, sup, (len(matched) + len(sup)) / total))

    # Supply voltage first (it explains current / power / heat that follow), then the best-evidenced cause, then priority.
    cands.sort(key=lambda c: (0 if c[0]["priority"] == 1 else 1, 1 if c[0].get("single_of") else 0, -c[3], c[0]["priority"]))
    chosen, covered_all = [], set()
    for p, matched, sup, score in cands:
        covers = {k for k, _ in matched + sup} | {k for alts in p['need'] for k, d in alts if dirs.get(k) == d}
        if chosen and covers <= covered_all:
            continue                                  # explains nothing new
        chosen.append((p, matched, sup, score, covers))
        covered_all |= covers
        if len(chosen) == 3:
            break

    used = set()                                      # every action text appears once in the whole plan
    hyps = []
    for n, (p, matched, sup, score, covers) in enumerate(chosen):
        order = [o["key"] for o in odd if o["key"] in covers]
        sev = max((odd_by_key[k]["severity"] for k in order), key=lambda s: SEV_RANK.get(s, 0))
        actions = [a for a in p["actions"] if a not in used]
        used.update(actions)
        if not actions:
            actions = ["Same checks as the step above."]
        conf = "Most likely cause" if n == 0 and score >= 0.5 else ("Likely cause" if score >= 0.5 else "Possible cause")
        if n > 0 and conf == "Most likely cause":
            conf = "Also possible"
        hyps.append({
            "id": p["id"], "title": p["title"], "short": p["short"], "confidence": conf, "urgent": p["urgent"],
            "cause": p["cause"], "risk": p["risk"], "because": "; ".join(_evidence(order, odd_by_key)),
            "actions": actions, "check": list(p["check"]), "call": p["call"], "severity": sev, "covers": order,
            "lead": (matched[0][0] if matched else None),
        })
    unexplained = [o for o in odd if o["key"] not in covered_all]
    return hyps, unexplained
