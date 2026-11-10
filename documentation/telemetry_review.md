# Telemetry review (v3.0)

Written from reading the code, not from assumptions. The database itself was not inspected, so row counts and
sizes are not quoted; the Manage ACs page now shows the real numbers.

## There are two data streams, and they do different jobs

| | `data_gathered` (training data) | `telemetry` (live telemetry) |
|---|---|---|
| Written when | AC Setup is in "data gathering" mode for an AC | AC Setup is in "telemetry" (monitoring) mode |
| Has `ac_unit` column | yes, one stream per AC | **no**, all devices share one stream |
| Used by the ML model | **yes: the only training source** (`fetch_unit_df`, newest 20,000 rows per AC) | **no** |
| Used for graphs | per-AC graphs | dashboard "latest", real-time graphs, detailed/single graphs |
| Used for alerts | no | no |

## Your questions

**What is telemetry used for?** `telemetry` feeds the dashboard's latest values (`/telemetry/latest`), the real-time
graphs (`/telemetry`), the detailed and single graphs (`/telemetry/column`) and the daily energy figure (first row of the
day is the baseline). Nothing else reads it.

**Does the ML model use historical telemetry?** For *training*: only `data_gathered` rows of that AC. For *inference*:
no history at all. Each incoming reading is scored on its own (`score_reading`) the moment it arrives, before anything is
stored. The only history the ML side keeps is `ac_health` (one score per AC per minute, pruned at 45 days) for the
Model Health trend, and the model itself (which holds up to 1,500 normal readings for the distance check).

**Are telemetry records necessary for anomaly detection?** No. Alerts are decided from the incoming reading, in memory.
A deleted or empty `telemetry` table changes no alert and no model.

**Essential vs. unnecessary**
- Essential for ML: the 10 model features (`dht_temp`, `dht_humidity`, `vibration`, `ds18b20_temp1/2`, `pzem_voltage`,
  `pzem_current`, `pzem_power`, `pzem_frequency`, `pzem_power_factor`) in `data_gathered`.
- Needed for graphs only: the same fields in `telemetry`, plus `pzem_energy` (cumulative counter, deliberately not a model feature).
- Redundant: `dust_sensor` (sensor removed, always empty); `ac_status` / `ac_thermostat` repeated on every row although they
  change rarely; long stretches of practically identical readings (idle compressor, steady running).

**Is the system storing more than it needs?** The live table, yes: at one reading per 10 s per device it grows by ~8,640
rows per device per day with no limit, and `/telemetry` returns *every* row when no date range is given. The training data is
not excess: more rows is how a model learns an AC's full range of behaviour.

## What v3.0 changed (and what it deliberately did not)

1. **Skip unchanged live readings** (`telemetry_store.Deduper`). A reading inside a small band of the last stored one
   (e.g. 0.3 °C, 1 V, 0.05 A, 5 W) is not stored, but one row is stored at least every 30 s (configurable), on any real
   change, when a sensor goes offline/back, and on the first reading of each day. The skipped reading is still scored by
   the model and the limits first, so **alerting sees every reading**. Switch it off on Manage ACs if you want every row.
   How much this saves depends on how often your ACs sit steady; I could not measure it without your data.
2. **Optional retention** for live telemetry (default **off = keep forever**). Runs once a day; has a preview ("N rows would
   go") and a confirmation. It can only delete from `telemetry`.
3. **Never touched:** `data_gathered`, `ac_models`, thresholds, limits, alerts. Training data is never thinned.

## Not changed, worth knowing
- `telemetry` has no `ac_unit` column, so with several devices in monitoring mode their rows are interleaved in one stream.
  Adding a column would fix the graphs but changes your table structure, so it is left for you to decide.
- `GET /telemetry` and `GET /data_gathered` return whole tables to the browser. The real-time page only shows the last 10
  rows. Limiting these endpoints is a safe next step but changes API behaviour for the graph pages, so I left them.
- The `dust_sensor` column can be dropped from both tables whenever you like.

## Can it be simplified without hurting detection accuracy?
Live telemetry: yes, thinning and retention have no effect on models or alerts. Training data: keep it. What matters there
is coverage (cooling, idle, hot days, start-ups), not the raw row count; if you ever want to shrink it, thin near-duplicate
rows from long steady periods rather than deleting old data wholesale, and retrain afterwards to compare the false-alarm
rate on the Model Health page.
