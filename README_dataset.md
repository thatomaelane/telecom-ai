# Telecom 60-day dataset (500 sites x 24 h x 60 days)

Period: 2026-09-01 00:00 to 2026-10-30 23:00 (hourly). 720,000 rows, same 48 columns and order as the original file.

## Files
| File | What it is |
|---|---|
| telecom_synthetic_60d_500sites.parquet | Main dataset (load with pandas.read_parquet or DuckDB) |
| site_master.csv | Asset register per site: region, battery kWh, PV kWp, fuel tank L, genset start SOC. Use in the optimiser |
| events_ground_truth.csv | Injected events (cyber, theft, HVAC failure, outages, heatwaves...) for validating detectors. DO NOT use as model features |
| generate_dataset.py | Reproducible generator (seed 42). All assumptions are in the CONFIG block |

## Formulas (verified 100% on all rows)
- site_pue = site_total_power_kw / telecom_dc_load_kw
- site_total_energy_kwh = running total of site_total_power_kw since 00:00 each day (same semantics as original). For hourly energy use site_total_power_kw x 1 h. Never use the cumulative column as a feature or target.
- diesel_consumed_liters = genset kWh x specific consumption (0.32 L/kWh, +-8% per site)
- scope_1 = 2.68 x diesel_L
- scope_2 = 0.95 x grid import kWh, where grid import = (load - solar used) when grid is up, else 0
- renewable_energy_share_pct = 100 x min(solar, load) / load
- generator_run_hours = 1 in hours the genset runs (same as original), diesel > 0 only when power_source_active = GENERATOR

## Deliberate changes vs the original day-1 data
1. Grid categories now behave as named: mean outage h/day high 0.44, medium 1.0, low 2.3, poor 4.2. Regional outages hit many sites at once. grid_voltage_v = 0 during an outage, and sags precede about half of outages.
2. Customer experience reacts to congestion (throughput falls, latency and drops rise with PRB; weak SINR/RSRP hurts). Rural (low tier) sites have weaker signal.
3. 5G/3G equipment adds power; enclosure type changes cooling load; PUE range is now 1.11 to 1.86.
4. Diesel is 0.32 L/kWh (original was about 1.1 L/kWh, unrealistic). Scope 2 factor is 0.95 kg/kWh (ASSUMPTION, original 0.75). Change EF_GRID_KG_PER_KWH once the current Eskom factor is confirmed.
5. Tamper, BMS-fault and IDS anomaly rates are lower than the original so they behave like incidents, not noise. network_traffic_anomaly_flag is a noisy IDS alert (about 3.5% false alarms plus 80% detection of real incidents).
6. BMS FAULT freezes battery_soc_pct and battery_soh_pct at the last healthy value (realistic data-quality issue).
7. Weekly pattern (weekend dip for business-type sites, month-end bump, Heritage Day 24 Sep), 0.08%/day traffic growth, spring warming, two heatwaves.
8. timestamp is a real datetime (original was text).

## Known limitations
- Still synthetic: relationships are simplified. State this clearly in the submission.
- battery charging energy is not added to site_total_power_kw or Scope 2.
- Only 60 days: no long-term seasonality.
