# AirBnBrrr: AI-Powered HVAC IoT Monitoring System

AirBnBrrr is an end-to-end IoT monitoring and proactive fault detection system designed specifically for Window-Type Air Conditioners. It ingests real-time telemetry from a Raspberry Pi sensor array and processes the data through a FastAPI backend to detect anomalies, generate explainable reports, and dispatch automated SMS alerts.

## Key Features

*   **IoT Telemetry Ingestion**: Receives real-time data from a custom Raspberry Pi sensor suite, including temperature, humidity, vibration, power consumption, current and voltage.
*   **Hybrid Diagnostic Pipeline**: A robust, three-layer approach to anomaly detection:
    1.  **Machine Learning (one model per AC)**: Every AC unit learns its own normal behaviour from its data in Supabase. Each model has two checks: an Isolation Forest (unusual combinations of readings) and a distance-to-normal check (how far a reading is from anything the AC has done before, which catches slow drift). The alert threshold is recomputed at every retrain from that unit's own recent normal readings, and the models are stored in Supabase so they survive redeploys.
    2.  **Safety limits (optional)**: A rule engine that checks readings against per-AC limits (e.g., compressor overload, undervoltage). A separate, switchable alarm: it never replaces the model.
    3.  **Generative AI (LLM)**: Queries the ultra-fast **Cerebras Inference API** using the **`zai-glm-4.7`** model (customizable via the web UI) to generate contextual diagnoses and forecasting. The backend includes a robust sanitization layer to clean markdown code blocks (```json ... ```) from LLM outputs for reliable JSON parsing.
*   **Automated SMS Alerts**: Integrates with Traccar SMS Gateway to instantly notify maintenance staff when critical faults are detected.
*   **ntfy Push Notifications**: Every alert also sends ONE ntfy notification per AC that lists all the problems found (hard limits, rule-engine findings and what the model found odd).
*   **AC Manager**: Add, rename or delete ACs from the dashboard; deleting an AC also removes its data and model. ACs without a custom name are shown as AC1, AC2, ... by position, so the remaining ones move up.
*   **Responsive Web Dashboard**: A clean, user-friendly interface built with Tailwind CSS to visualize telemetry data, review detailed anomaly reports side-by-side, and manage system contacts.

## Technology Stack

*   **Backend**: Python, FastAPI
*   **Database**: PostgreSQL (Supabase)
*   **Machine Learning**: scikit-learn (Isolation Forest)
*   **Frontend**: HTML, Vanilla JavaScript, Tailwind CSS, Lucide Icons
*   **Integrations**: Traccar SMS Gateway, Generative AI APIs

## Installation and Setup

### Prerequisites
*   Python 3.9+
*   PostgreSQL database (or Supabase project)

### Environment Configuration
Create a `.env` file in the root directory with the following variables:
```env
SUPABASE_URL=postgresql://[user]:[password]@[host]:[port]/[db_name]
# Add any required API keys for the Generative AI service here
```

### Setup Steps

1.  **Clone the repository and enter the directory**:
    ```bash
    git clone https://github.com/gericandmorty/AirBnbrrr.git
    cd AirBnbrrr
    ```

2.  **Create and activate a virtual environment**:
    *   Windows:
        ```cmd
        python -m venv venv
        venv\Scripts\activate
        ```
    *   macOS/Linux:
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the models** (creates the extra tables the first time):
    ```bash
    python train_models.py
    ```
    You can also train a single AC with the **Train now** button on the **Manage ACs** page. Until an AC has a model (it needs at least 200 readings) it has no ML alerts; only the safety limits and sensor alerts cover it.

### Alerts: environment variables
| Variable | Needed | Meaning |
|---|---|---|
| `SUPABASE_URL` | yes | Postgres connection string |
| `TELEMETRY_API_KEY` | recommended | A long random text. When set, only devices that send the same text in the `X-API-Key` header can post readings. The ESP32 reads it from `esp32/config.py` |
| `NTFY_TOPIC` | for ntfy | Use a long random name, e.g. `AirBnrr_alien_test_code_k8x2m9qv4t`. Public ntfy topics can be read by anyone who knows the name |
| `APP_URL` | no | Dashboard link opened when you tap the notification |
| `NTFY_SERVER` | no | Only if you self-host ntfy (default `https://ntfy.sh`) |
| `NTFY_TOKEN` | no | Access token for a protected topic |

Subscribe to the same topic in the ntfy phone app, then open `/api/ntfy/test` while logged in to send a test notification.

### ESP32 (the `esp32/` folder)
The ESP32 can post straight to your Render site over any WiFi with internet; the Raspberry Pi is not needed as a go-between. Edit `esp32/config.py` (WiFi, `SERVER_URL`, `API_KEY`, `AC_UNIT`), upload the files with Thonny, and set `TELEMETRY_API_KEY` on Render to the same text as `API_KEY`. A sensor that stops answering is sent as empty (null), which the site reports as "sensor offline".

### How alerts work (v3.1): three independent channels
Every reading is judged by three channels that share nothing, so one can never delay or suppress another. Each alert says where it comes from (badge on the Alerts page, `[ML]` / `[SAFETY]` / `[SENSOR]` in SMS, `(ML)` / `(Safety)` / `(Sensor)` in the ntfy title).

| Channel | Decided by | Switch | Timing |
|---|---|---|---|
| **ML** | the trained model of *that* AC (Isolation Forest + distance-to-normal). Alerts even when every reading is inside the safety limits | none, always on. No model (never trained, data deleted, unreadable) = no ML alert, never a threshold dressed up as ML | **INSTANT** when the finding is severe on **two readings in a row** (about 10 s): a Critical departure from normal, or current / voltage / power far above this AC's normal. Only for models trained with the current version (retrain old ones with Train now). Everything else must still be abnormal 3 minutes later |
| **Safety** | the per-AC limits on Manage ACs | **Safety Limit Alerts** on the Dashboard | **INSTANT:** current / voltage / power over the limit, and vibration or a compressor temperature over the limit while the model also says the AC is abnormal. Every other limit: 3 minutes |
| **Sensor** | a sensor that did not answer | none | 3 minutes |

**Instant means instant.** The alert decision is made from memory on the reading that causes it, *before* the reading is stored: a slow or failing database can no longer delay an alert, and the SMS no longer depends on the database either (limits, settings and the AC setup are read from a cache that refreshes in the background; the database connection has a 10 s timeout). The log shows `[LATENCY]` lines (reading arrived -> decided -> ntfy sent -> SMS handed to the gateway) so you can see where any remaining time goes. Power factor has no safety limit, so it only appears in ML alerts. An instant alert fires on the first reading that meets its condition. It only waits for the previous *instant* alert of the same channel (2 minutes, so it does not repeat every 10 s); a normal alert sent a moment earlier never holds it back. Each alert is its own thread: ntfy goes out at once, the alert is saved and the SMS sent next, and the AI opinion (slow) is fetched last and attached to the alert afterwards. Normal alerts keep their 3-minute confirmation (`CRITICAL_REPEAT_S`, `SOFT_DELAY_S`, `STILL_WINDOW_S` in `services/alert_channels.py`).

**Probable causes, not canned advice (v3.5).** The model says which sensors are off and in which direction; `services/ml_diagnosis.py` then reads the *combination* and names 1 to 3 probable causes (for example: supply voltage too high, compressor running too hot, low refrigerant, weak airflow / coil icing, compressor not running, loose mounts or a worn bearing), each with the readings that support it. Every action appears only once in a plan, and advice only appears where it is a likely cause (the air filter is suggested for a freezing coil, not for an over-voltage).

**What an ML alert contains.** Severity (Low / Medium / High / Critical) and the AC condition (Need Repair / Check Power Supply / Need Maintenance), the odd sensors with their real values and this AC's usual range, why the model calls it abnormal (in plain words, with the numbers), what could go wrong, and what to do. It is built by `services/ml_explain.py` from what the model measured: how far past normal the reading is (`distance_ratio`), which check fired (unusual combination and/or distance from normal readings), and each sensor's deviation, compared with the AC's usual values *while running* or *while idle*. Severity is not a safety limit: it is calibrated on fault data (`tests/fault_data.py`). The Dashboard "ML Detection" panel and the Alerts page health badge use the same assessment, so an abnormal AC is never shown as "Good condition" just because no limit was crossed.

A problem both channels see sends two messages. High humidity is no longer an alert anywhere; only **low** humidity is (a safety limit). Humidity is still shown and still one of the model's inputs.
A compressor that is off between cooling cycles is normal (non-inverter AC): the rule engine skips the "running" checks while the current is under 1 A.

### Phone notifications (ntfy) not arriving?
The Dashboard (Alert settings) shows whether ntfy works and, if the last one failed, the exact reason; **Send test** sends a notification now. Common causes: `NTFY_TOPIC` missing on the server; `NTFY_SERVER` written without `https://` (now fixed automatically); a quoted or space-padded topic (now cleaned); rate limiting by ntfy.sh (HTTP 429, shared hosting IPs: set `NTFY_TOKEN` or self-host); a protected topic (HTTP 401/403: set `NTFY_TOKEN`). Failed sends are retried and logged as `[NTFY] ...` in the server log. `GET /api/ntfy/status` returns the same information.

### Delete Data (Manage ACs)
Removes the stored training readings of ONE AC and the model learned from them (the model keeps copies of normal readings). The AC, its name, its limits, live telemetry and past alerts stay. The AC then shows **retraining required** and has no ML detection until 200+ new readings are collected and **Train now** is pressed.

### Tests
`python tests/test_ml_alerting.py` trains real per-AC models and runs normal readings and ten generated fault datasets through real inference (needs pandas, numpy, scikit-learn only). To check the model on your own data: `python tools/check_model.py normal.csv [abnormal.csv]` (Export CSV on Manage ACs).

### Version
The file `VERSION` holds the version (shown in the sidebar and at `/api/version`), so you can always tell which code is deployed. Downloads are named `Mochi_Lil_vX_Y.zip`.

### Per-AC models, AC manager and alerts
*   **Device code vs label**: the Pi/ESP32 sends a fixed `ac_unit` (the device code, e.g. `AC3`). The dashboard shows each AC by its position in the list (AC1, AC2, ...), so deleting one moves the others up without touching any stored data.
*   **Safety limits are per AC and editable on the web**: Manage ACs -> Safety limits. A new AC starts with the default limits (the original 1.5 HP numbers). Current, voltage and power (high) limits always alert instantly. Vibration and the two compressor temperature limits only make an alert critical when the model has already flagged the AC. The rule engine, the alert texts and the readings panel all use the AC's own limits.
*   Pages: `/pages/ac_manager` (add, delete, train, export CSV) and `/pages/ml_learning` (Model Health: how normal each AC looks over time, what it learned, how it learned, and an early-warning trend).

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The web dashboard will be available at `http://127.0.0.1:8000`.

## Architecture Overview

1.  **Raspberry Pi** collects sensor data and sends a POST request to `/telemetry`.
2.  The backend stores the raw data and scores it with that AC's own **Isolation Forest** model.
3.  If an anomaly is flagged, the data is passed to the **Rule Engine** for physics-based threshold verification.
4.  The system then queries the **LLM** for additional contextual analysis.
5.  A combined JSON report is saved, an **SMS alert** with a summary is sent to authorized contacts, and one **ntfy** notification lists every problem found on that AC.


## Try it without waiting for a real fault: the simulator
`tools/simulate_ac.py` sends simulated live readings to your site: a normal AC, a slow drift that stays inside every safety limit (only the model notices it, "Check" alert about 8 minutes later), then a real fault ("STOP" alert at once), then recovery. It needs only Python 3.

1. Add the training data of a demo AC (or use any AC that has a trained model) and press **Train now** on Manage ACs.
2. In **AC Setup** use monitoring (telemetry) mode, never data gathering: the script refuses to run otherwise, so fake readings cannot become training data.
3. `python tools/simulate_ac.py` (it sends to the address in `DEFAULT_URL` at the top of the script; add `--key YOUR_TELEMETRY_API_KEY` if you set one).
4. At the end it prints the SQL that removes the fake readings, health points and alerts.
