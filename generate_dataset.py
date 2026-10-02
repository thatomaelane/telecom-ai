#!/usr/bin/env python3
"""
60-day synthetic telecom dataset generator (500 sites x 24 h x 60 days = 720,000 rows).

Anchors on the team's original 1-day dataset (static site attributes, site demand scale,
signal quality, initial battery/fuel state) and simulates physically-coupled behaviour on top.

Usage:
    python generate_dataset.py --input telecom_synthetic_dataset_12000x48.xlsx --outdir out --days 60

All tunable assumptions are in the CONFIG block. Change EF_GRID_KG_PER_KWH to the
current Eskom factor before final ESG numbers are quoted.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- CONFIG
START = "2026-09-01"
SEED = 42
EF_GRID_KG_PER_KWH = 0.95      # Scope 2 factor (kg CO2e / kWh). ASSUMPTION - verify vs latest Eskom factor.
EF_DIESEL_KG_PER_L = 2.68      # Scope 1 factor (kg CO2e / litre diesel) - same as original dataset.
SFC_L_PER_KWH = 0.32           # Diesel genset specific consumption (L per kWh generated), site-varied +-10 %.
HOLIDAYS = {"2026-09-24"}      # Heritage Day (Thursday)
PAYDAY_DAYS = {25, 26, 27}     # month-end data-usage bump

CAT_OUTAGE_HOURS = {"high": 0.35, "medium": 0.9, "low": 2.3, "poor": 4.5}   # mean local outage h/day
CAT_OUTAGE_DUR = {"high": 1.4, "medium": 1.8, "low": 2.4, "poor": 3.0}      # mean block length (h)
CAT_REGIONAL_HIT = {"high": 0.35, "medium": 0.6, "low": 0.85, "poor": 0.95}
CAT_VOLT_SD = {"high": 3.0, "medium": 3.5, "low": 4.5, "poor": 6.0}
CAT_EXPECTED_AVAIL = {"high": 23.65, "medium": 23.1, "low": 21.8, "poor": 19.5}

COLS = ["timestamp", "site_id", "data_volume_gb", "prb_utilization_pct", "active_rrc_connected_users",
        "busy_hour_traffic_ratio", "site_total_power_kw", "site_total_energy_kwh", "telecom_dc_load_kw",
        "site_pue", "grid_voltage_v", "power_source_active", "grid_availability_hours_daily",
        "battery_soc_pct", "battery_soh_pct", "solar_generation_kw", "generator_fuel_level_pct",
        "generator_run_hours", "cabinet_internal_temp_c", "battery_bank_temp_c", "ambient_outdoor_temp_c",
        "hvac_operational_status", "cooling_fan_speed_rpm", "failed_login_attempts_count",
        "network_traffic_anomaly_flag", "physical_tamper_sensor_status", "unauthorized_ip_scan_count",
        "bms_modbus_comm_status", "diesel_consumed_liters", "scope_1_carbon_kg_co2e",
        "scope_2_carbon_kg_co2e", "renewable_energy_share_pct", "battery_recycling_status",
        "call_setup_success_rate_pct", "call_drop_rate_pct", "avg_dl_throughput_mbps",
        "avg_ul_throughput_mbps", "latency_rtt_ms", "rsrp_dbm", "sinr_db", "site_demographic_tier",
        "tower_structure_type", "enclosure_type", "backhaul_technology", "deployed_technologies",
        "grid_reliability_category", "latitude", "longitude"]
STATIC = ["site_demographic_tier", "tower_structure_type", "enclosure_type", "backhaul_technology",
          "deployed_technologies", "grid_reliability_category", "latitude", "longitude"]


def ar1(rng, shape, phi, sd):
    """AR(1) noise along the last axis with stationary std = sd."""
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), shape)
    out = np.empty(shape)
    out[..., 0] = rng.normal(0, sd, shape[:-1])
    for i in range(1, shape[-1]):
        out[..., i] = phi * out[..., i - 1] + e[..., i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ site master from day-1 data
    raw = pd.read_excel(a.input)
    g = raw.groupby("site_id", sort=True)
    sm = g[STATIC].first()
    sm["battery_soh_pct0"] = g.battery_soh_pct.first()
    sm["vol_mean"] = g.data_volume_gb.mean()
    sm["users_per_gb"] = (raw.active_rrc_connected_users / raw.data_volume_gb).groupby(raw.site_id).mean()
    sm["sinr_mean"] = g.sinr_db.mean()
    sm["rsrp_mean"] = g.rsrp_dbm.mean()
    sm["soc0"] = g.battery_soc_pct.first()
    sm["fuel0"] = g.generator_fuel_level_pct.first()
    sm["solar_peak_proxy"] = g.solar_generation_kw.max()
    sm["recycling0"] = g.battery_recycling_status.first()
    for c in sm.columns:  # pandas>=3 string dtype -> plain numpy object arrays
        if not pd.api.types.is_numeric_dtype(sm[c]):
            sm[c] = sm[c].astype(object)
    sites = np.asarray(sm.index, dtype=object)
    S, D = len(sites), a.days
    T = D * 24

    ts = pd.date_range(START, periods=T, freq="h")
    hour = np.tile(np.arange(24), D)
    hh = hour + 0.5
    day = np.repeat(np.arange(D), 24)
    dow = ts.dayofweek.values
    dom = ts.day.values
    is_weekend = dow >= 5
    is_holiday = np.array([d.strftime("%Y-%m-%d") in HOLIDAYS for d in ts])
    events = []  # ground-truth log

    def log(site_idx, t0, dur, typ, detail=""):
        t1 = min(t0 + dur, T) - 1
        events.append(dict(site_id=sites[site_idx] if site_idx is not None else "", start=ts[t0], end=ts[t1],
                           hours=int(t1 - t0 + 1), event_type=typ, detail=detail))

    lat, lon = sm.latitude.values, sm.longitude.values
    tier, tech = sm.site_demographic_tier.values, sm.deployed_technologies.values
    encl, bh, cat = sm.enclosure_type.values, sm.backhaul_technology.values, sm.grid_reliability_category.values

    # regions (2.5 deg cells) share weather, outages and scan campaigns
    rkey = (np.floor((lat + 35) / 2.5) * 100 + np.floor((lon - 16) / 2.5)).astype(int)
    _, region = np.unique(rkey, return_inverse=True)
    R = region.max() + 1

    # ------------------------------------------------------------------ regional weather
    temp_anom = ar1(rng, (R, D), 0.7, 2.2)
    for _ in range(2):  # two heatwaves
        d0 = int(rng.integers(8, D - 6))
        regs = rng.choice(R, size=max(1, int(0.3 * R)), replace=False)
        temp_anom[regs, d0:d0 + 3] += 5.5
        for r in regs:
            log(None, d0 * 24, 72, "heatwave", f"region {r}, +5.5C anomaly")
    cloud_day = np.clip(rng.beta(6, 2, (R, D)), 0.15, 1.0)

    # ------------------------------------------------------------------ ambient temperature
    alt_off = rng.normal(0, 1.8, S)
    amp = rng.normal(7.0, 1.0, S)
    site_mean = 17.5 + 0.55 * (lat + 28.3) + alt_off
    x = (hh - 5.5) / 9
    y = ((hh - 14.5) % 24) / 15
    shape = np.where((hh >= 5.5) & (hh <= 14.5), -np.cos(np.pi * x), np.cos(np.pi * y))
    amb = (site_mean[:, None] + 0.055 * (np.arange(T) / 24)[None, :] + temp_anom[region][:, day]
           + amp[:, None] * shape[None, :] + ar1(rng, (S, T), 0.8, 0.6))
    amb = np.clip(amb, 0.0, 44.0)

    # ------------------------------------------------------------------ solar PV
    kwp = sm.solar_peak_proxy.values * 1.15
    sr = 6.35 - 0.012 * np.arange(T) / 24
    ss = 17.75 + 0.011 * np.arange(T) / 24
    frac = (hh - sr) / (ss - sr)
    elev = np.where((frac > 0) & (frac < 1), np.sin(np.pi * np.clip(frac, 0, 1)) ** 1.15, 0.0)
    cloud = np.clip(cloud_day[region][:, day] + ar1(rng, (S, T), 0.7, 0.10), 0.08, 1.0)
    solar = kwp[:, None] * elev[None, :] * cloud * 0.92

    # ------------------------------------------------------------------ cyber incidents (drive traffic too)
    flood_mult = np.ones((S, T))
    users_mult = np.ones((S, T))
    login_mult = np.ones((S, T))
    scan_mult = np.ones((S, T))
    incident = np.zeros((S, T), bool)
    bms_att = np.zeros((S, T), bool)
    bh_mult = {"fibre": 1.0, "hybrid": 1.2, "microwave": 1.4}
    exposure = np.array([bh_mult[b] for b in bh])
    tier_theft = np.array([{"high": 0.7, "medium": 1.0, "low": 1.6}[t] for t in tier])
    p_inc = {"brute_force": 0.010, "port_scan": 0.012, "traffic_flood": 0.006, "modbus_intrusion": 0.003}
    for typ, p in p_inc.items():
        hit = rng.random((S, D)) < p * exposure[:, None]
        for s, d in zip(*np.nonzero(hit)):
            add_incident = None
            if typ == "brute_force":
                dur = int(rng.integers(2, 7)); st = int(rng.choice([0, 1, 2, 3, 22, 23, 9, 14, 19]))
                t0 = d * 24 + st; login_mult[s, t0:t0 + dur] *= rng.uniform(8, 15)
            elif typ == "port_scan":
                dur = int(rng.integers(3, 11)); st = int(rng.integers(0, 24))
                t0 = d * 24 + st; scan_mult[s, t0:t0 + dur] *= rng.uniform(6, 12)
            elif typ == "traffic_flood":
                dur = int(rng.integers(2, 6)); st = int(rng.integers(0, 24))
                t0 = d * 24 + st; f = rng.uniform(1.8, 3.2)
                flood_mult[s, t0:t0 + dur] *= f; users_mult[s, t0:t0 + dur] *= 1.3
                scan_mult[s, t0:t0 + dur] *= 2.0
            else:
                dur = int(rng.integers(1, 5)); st = int(rng.integers(0, 24))
                t0 = d * 24 + st; login_mult[s, t0:t0 + dur] *= 3.0; bms_att[s, t0:t0 + dur] = True
            incident[s, t0:t0 + dur] = True
            log(s, t0, dur, typ)
    # regional coordinated scan campaigns
    for r in range(R):
        for d in np.nonzero(rng.random(D) < 0.02)[0]:
            st = int(rng.integers(0, 18)); members = np.nonzero(region == r)[0]
            for s in members[rng.random(len(members)) < 0.35]:
                dur = int(rng.integers(3, 8)); t0 = d * 24 + st
                scan_mult[s, t0:t0 + dur] *= rng.uniform(6, 10); incident[s, t0:t0 + dur] = True
                log(s, t0, dur, "regional_scan_campaign", f"region {r}")

    # ------------------------------------------------------------------ physical: tamper / theft / maintenance
    tamper = np.zeros((S, T), bool)
    fuel_theft = np.zeros((S, T))       # % points of tank stolen at hour t
    cable_theft = np.zeros((S, T), bool)
    soh_step = np.zeros((S, T))         # cumulative SOH loss (applied at t onward)
    for s, d in zip(*np.nonzero(rng.random((S, D)) < 0.05)):     # authorised maintenance visits
        t0 = d * 24 + int(rng.integers(8, 16)); tamper[s, t0:t0 + int(rng.integers(1, 3))] = True
    tamper |= rng.random((S, T)) < 0.0004                          # sensor false alarms
    for s, d in zip(*np.nonzero(rng.random((S, D)) < 0.0025 * tier_theft[:, None])):
        st = int(rng.choice([22, 23, 0, 1, 2, 3, 4])); t0 = d * 24 + st; dur = int(rng.integers(1, 4))
        tamper[s, t0:t0 + dur] = True
        if rng.random() < 0.5:
            fuel_theft[s, t0] = rng.uniform(15, 40); log(s, t0, dur, "fuel_theft")
        else:
            cable_theft[s, t0:t0 + dur + 2] = True; soh_step[s, t0:] += rng.uniform(3, 8)
            log(s, t0, dur + 2, "battery_cable_theft")

    # ------------------------------------------------------------------ traffic demand
    res = np.array([6.56, 6.64, 6.71, 6.69, 7.27, 8.49, 10.02, 11.26, 11.78, 11.16, 9.84, 8.51,
                    7.45, 7.67, 8.26, 10.02, 13.24, 16.97, 20.56, 21.87, 19.99, 17.0, 13.46, 10.15])
    res = res / res.mean()
    bus = 0.35 + 0.9 * np.exp(-((np.arange(24) - 11.5) / 3.5) ** 2) + 0.5 * np.exp(-((np.arange(24) - 15) / 2.5) ** 2)
    bus = bus / bus.mean()
    w = rng.beta(1.2, 1.8, S)                                     # business-profile weight per site
    prof = (1 - w)[:, None] * res[None, :] + w[:, None] * bus[None, :]
    prof = prof / prof.mean(axis=1, keepdims=True)
    scale = sm.vol_mean.values
    vpk = scale * prof.max(axis=1)
    wk = np.where(is_weekend | is_holiday, 1.0, 0.0)
    wk_fac = 1 + wk[None, :] * (-0.35 * w[:, None] + 0.12 * (1 - w[:, None]))
    payday = np.isin(dom, list(PAYDAY_DAYS)).astype(float)[None, :] * 0.06 + 1
    dayf = np.repeat(rng.lognormal(0, 0.06, (S, D)), 24, axis=1)
    growth = (1 + 0.0008 * np.arange(T) / 24)[None, :]
    ev_mult = np.ones((S, T))
    for s, d in zip(*np.nonzero(rng.random((S, D)) < 0.006)):
        dur = int(rng.integers(3, 6)); t0 = d * 24 + int(rng.integers(16, 20))
        ev_mult[s, t0:t0 + dur] *= rng.uniform(1.6, 2.5); log(s, t0, dur, "local_traffic_event")
    vol = (scale[:, None] * prof[:, hour] * wk_fac * payday * dayf * growth * ev_mult * flood_mult
           * np.exp(ar1(rng, (S, T), 0.5, 0.13)))
    vol = np.clip(vol, 1.0, 130.0)
    u = vol / vpk[:, None]
    rho = rng.lognormal(0, 0.22, S)
    prb = np.clip(12 + 52 * u * rho[:, None] + rng.normal(0, 2.5, (S, T)), 5, 98)
    users = vol * sm.users_per_gb.values[:, None] * users_mult * np.exp(rng.normal(0, 0.07, (S, T)))
    busy = np.clip(0.45 + 0.55 * np.minimum(u, 1.0) + rng.normal(0, 0.02, (S, T)), 0.45, 1.0)

    # ------------------------------------------------------------------ energy: DC load, cooling, total
    tech_kw = np.array([{"4G": 0.0, "3G_4G": 0.4, "4G_5G": 0.8, "3G_4G_5G": 1.1}[t] for t in tech])
    dc_base = 3.3 + tech_kw + rng.normal(0, 0.3, S)
    dc = dc_base[:, None] + 0.05 * prb + rng.normal(0, 0.12, (S, T))
    encl_off = np.array([{"shelter": 0.0, "container": 1.5, "outdoor": 3.0}[e] for e in encl])
    hvac_kw_s = np.array([{"shelter": 2.2, "container": 1.7, "outdoor": 1.1}[e] for e in encl])
    raw_temp = 7.5 + 0.93 * amb + 0.04 * dc + encl_off[:, None]
    heat_day = temp_anom[region][:, day] > 4
    hvac_fail = np.zeros((S, T), bool)
    p_hf = 0.004 * np.where(heat_day[:, ::24], 4.0, 1.0)
    for s, d in zip(*np.nonzero(rng.random((S, D)) < p_hf)):
        t0 = d * 24 + int(rng.integers(0, 24)); dur = int(rng.integers(8, 31))
        hvac_fail[s, t0:t0 + dur] = True; log(s, t0, dur, "hvac_failure")
    hvac_on = (raw_temp > 28 + rng.normal(0, 0.8, (S, T))) & ~hvac_fail
    cab = np.where(hvac_on, 25 + 0.3 * (raw_temp - 28), raw_temp) + rng.normal(0, 0.5, (S, T))
    rpm = np.clip(570 + 157 * np.maximum(cab - 17.5, 0) + rng.normal(0, 70, (S, T)), 570, 5000).round()
    bank = 2.0 + 0.55 * amb + 0.35 * cab + rng.normal(0, 0.7, (S, T))
    hvac_kw = hvac_kw_s[:, None] * np.clip(0.6 + 0.05 * (raw_temp - 28), 0.6, 1.8) * hvac_on
    fan_kw = 0.35 * (rpm / 3000) ** 3
    load = dc * 1.12 + 0.15 + hvac_kw + fan_kw + rng.normal(0, 0.05, (S, T))  # rectifier/battery-charger losses + lighting/security
    pue = load / dc

    # ------------------------------------------------------------------ grid outages
    w_start = np.full(24, 0.6); w_start[5:10] = 2.0; w_start[16:22] = 2.5; w_start /= w_start.sum()
    outage = np.zeros((S, T), bool)
    brown = np.zeros((S, T), bool)
    for s in range(S):
        c = cat[s]; n = rng.poisson(CAT_OUTAGE_HOURS[c] / CAT_OUTAGE_DUR[c], D)
        for d in np.nonzero(n)[0]:
            for _ in range(n[d]):
                dur = int(np.clip(round(rng.gamma(2.0, CAT_OUTAGE_DUR[c] / 2.0)), 1, 8))
                t0 = d * 24 + int(rng.choice(24, p=w_start))
                outage[s, t0:t0 + dur] = True
                if rng.random() < 0.5 and t0 > 0:
                    brown[s, t0 - 1] = True
    for r in range(R):
        members = np.nonzero(region == r)[0]
        for d in np.nonzero(rng.random(D) < 0.025)[0]:
            t0 = d * 24 + int(rng.integers(0, 24)); dur = int(rng.integers(3, 10)); n_hit = 0
            for s in members:
                if rng.random() < CAT_REGIONAL_HIT[cat[s]]:
                    outage[s, t0:t0 + dur] = True; n_hit += 1
            if n_hit:
                log(None, t0, dur, "regional_grid_outage", f"region {r}, {n_hit} sites")
    grid_up = ~outage
    volt_sd = np.array([CAT_VOLT_SD[c] for c in cat])
    volt = 230 + rng.normal(0, 1, (S, T)) * volt_sd[:, None]
    volt = np.where(brown, volt - rng.uniform(10, 25, (S, T)), volt)
    volt = np.where(outage, 0.0, np.clip(volt, 190, 255))
    exp_avail = np.array([CAT_EXPECTED_AVAIL[c] for c in cat])
    cs = np.cumsum(grid_up, axis=1)
    prev = np.zeros((S, T)); prev[:, 24:] = cs[:, :-24]
    avail = cs - prev
    early = np.arange(T) < 24
    avail = np.where(early[None, :], cs + (23 - np.arange(T))[None, :] * (exp_avail[:, None] / 24), avail)

    # ------------------------------------------------------------------ power-source dispatch (sequential)
    cap_kwh = rng.lognormal(np.log(38), 0.3, S) * np.where(cat == "poor", 1.25, 1.0)
    tank_l = rng.uniform(250, 600, S)
    gen_thr = rng.uniform(45, 60, S)
    float_soc = np.clip(rng.normal(93, 4, S), 82, 100)
    sfc = SFC_L_PER_KWH * rng.normal(1.0, 0.08, S)
    soh_rate = rng.uniform(0.006, 0.02, S)
    soh = sm.battery_soh_pct0.values[:, None] - soh_rate[:, None] * (np.arange(T) / 24)[None, :] - np.cumsum(soh_step * 0, axis=1)
    soh = soh - soh_step
    replaced = np.zeros((S, T), bool)
    for s in np.nonzero(sm.battery_soh_pct0.values < 79)[0]:
        if rng.random() < 0.35:
            tr = int(rng.integers(10, D - 5)) * 24
            soh[s, tr:] = rng.uniform(96, 99) - soh_rate[s] * (np.arange(T - tr) / 24)
            replaced[s, tr:] = True; log(s, tr, 1, "battery_replacement")

    soc = sm.soc0.values.copy().astype(float)
    fuel = sm.fuel0.values.copy().astype(float)
    gen_on = np.zeros(S, bool)
    SRC = np.empty((S, T), dtype="<U9"); SOC = np.empty((S, T)); FUEL = np.empty((S, T))
    DIESEL = np.zeros((S, T)); GENRUN = np.zeros((S, T), int); DEGR = np.zeros((S, T), bool)
    for t in range(T):
        up = grid_up[:, t]; sol = solar[:, t]; ld = load[:, t]
        sol_used = np.minimum(sol, ld); net = ld - sol_used
        fuel = np.maximum(fuel - fuel_theft[:, t], 0)
        if hour[t] == 9:
            low = (fuel < 30) & (rng.random(S) < 0.6)
            fuel = np.where(low, rng.uniform(88, 100, S), fuel)
        gen_on &= ~up
        gen_on |= (~up) & (soc <= gen_thr) & (fuel > 2.5)
        gen_on &= (fuel > 2.5) & (soc < 88)
        eff_cap = cap_kwh * soh[:, t] / 100
        chg_grid = np.where(soc < 80, 9.0, 3.0)
        soc_up = np.where(soc < float_soc, np.minimum(soc + chg_grid, float_soc), soc)
        soc_gen = np.minimum(soc + 7.0, 90.0)
        drain = net / eff_cap * 100 / 0.95
        soc_batt = np.maximum(soc - drain, 5.0)
        soc = np.where(up, soc_up, np.where(gen_on, soc_gen, soc_batt))
        soc = np.where(cable_theft[:, t], np.minimum(soc, 20.0), soc)
        gen_kwh = np.where(gen_on, net + 0.07 * cap_kwh, 0.0)
        diesel = gen_kwh * sfc * (1 + 0.3 * np.exp(-net / 4))
        fuel = np.maximum(fuel - diesel / tank_l * 100, 0)
        solar_src = sol >= 0.6 * ld
        src = np.where(gen_on, "GENERATOR", np.where(solar_src, "SOLAR", np.where(up, "GRID", "BATTERY")))
        SRC[:, t] = src; SOC[:, t] = soc; FUEL[:, t] = fuel; DIESEL[:, t] = diesel
        GENRUN[:, t] = gen_on.astype(int); DEGR[:, t] = (~up) & (~gen_on) & (soc <= 15)
    sol_used_all = np.minimum(solar, load)
    net_all = load - sol_used_all
    grid_import = np.where(grid_up, net_all, 0.0)
    scope2 = EF_GRID_KG_PER_KWH * grid_import
    scope1 = EF_DIESEL_KG_PER_L * DIESEL
    renew = 100 * sol_used_all / load

    # ------------------------------------------------------------------ BMS / cyber signals
    p_bms = 0.0015 + 0.02 * (bank > 38)
    bms_fault = (rng.random((S, T)) < p_bms) | bms_att | cable_theft
    bms_fault_hold = bms_fault.copy()
    # frozen readings: battery SOC/SOH hold last healthy value while comms are down
    idx = np.where(~bms_fault, np.arange(T)[None, :], 0); np.maximum.accumulate(idx, axis=1, out=idx)
    SOC_rep = np.take_along_axis(SOC, idx, axis=1)
    soh_rep = np.take_along_axis(soh, idx, axis=1)
    lam_login = rng.lognormal(np.log(3.5), 0.6, S)
    lam_scan = rng.lognormal(np.log(2.4), 0.6, S)
    diurnal_bot = 1 + 0.25 * np.cos(2 * np.pi * (hour - 2) / 24)
    logins = rng.poisson(lam_login[:, None] * diurnal_bot[None, :] * login_mult)
    scans = rng.poisson(lam_scan[:, None] * scan_mult)
    ids_tp = incident & (rng.random((S, T)) < 0.8)
    ids_fp = rng.random((S, T)) < 0.035
    anomaly = (ids_tp | ids_fp | (flood_mult > 1.5)).astype(int)

    # ------------------------------------------------------------------ customer experience
    tier_sinr = np.array([{"high": 1.0, "medium": 0.0, "low": -1.5}[t] for t in tier])
    tier_rsrp = np.array([{"high": 2.5, "medium": 0.0, "low": -4.0}[t] for t in tier])
    site_sinr = sm.sinr_mean.values + tier_sinr + rng.normal(0, 1.2, S)
    site_rsrp = sm.rsrp_mean.values + tier_rsrp + rng.normal(0, 2.0, S)
    rsrp = np.clip(site_rsrp[:, None] + ar1(rng, (S, T), 0.7, 2.2), -120, -60)
    sinr = np.clip(site_sinr[:, None] + ar1(rng, (S, T), 0.5, 1.8) - 0.04 * (prb - 40), 0, 35)
    peak = np.array([{"4G": 70, "3G_4G": 66, "4G_5G": 92, "3G_4G_5G": 92}[x] for x in tech]) * rng.lognormal(0, 0.10, S)
    bh_f = np.array([{"fibre": 1.0, "hybrid": 0.95, "microwave": 0.9}[x] for x in bh])
    tier_f = np.array([{"high": 1.05, "medium": 1.0, "low": 0.9}[x] for x in tier])
    sinr_f = 0.4 + 0.6 / (1 + np.exp(-(sinr - 10) / 5))
    load_f = 1 - 0.65 * (prb / 100) ** 1.3
    hot = cab > 45
    degr = DEGR | hot
    dl = peak[:, None] * bh_f[:, None] * tier_f[:, None] * sinr_f * load_f * rng.lognormal(0, 0.05, (S, T))
    dl = dl * np.where(degr, 0.5, 1.0)
    ul = dl * 0.27 * rng.lognormal(0, 0.06, (S, T))
    lat_bh = np.array([{"fibre": 0.0, "hybrid": 2.0, "microwave": 5.0}[x] for x in bh])
    lat_tier = np.array([{"high": 0.0, "medium": 0.0, "low": 2.0}[x] for x in tier])
    latency = ((9 + lat_bh + lat_tier)[:, None] + 3.5 * (prb / (100.5 - prb)) ** 0.8) * rng.lognormal(0, 0.12, (S, T))
    latency = latency + np.where(degr, 15, 0)
    c = np.clip((prb - 65) / 33, 0, 1)
    weak = np.maximum(12 - sinr, 0)
    rsrp_pen = np.maximum(-95 - rsrp, 0)
    drop = np.clip(0.45 + 0.65 * c ** 1.5 + 0.045 * weak + 0.02 * rsrp_pen + rng.normal(0, 0.07, (S, T)), 0.05, None)
    drop = drop + np.where(degr, 1.5, 0)
    setup = np.clip(99.65 - 1.4 * c - 0.06 * weak - 0.02 * rsrp_pen + rng.normal(0, 0.15, (S, T)), 80, 100)
    setup = setup - np.where(degr, 6, 0)

    # ------------------------------------------------------------------ battery recycling status
    status = np.tile(sm.recycling0.values[:, None], (1, T)).astype(object)
    review_day = np.where(rng.random(S) < 0.5, rng.integers(0, D, S), 10 ** 6)
    for s in range(S):
        if sm.recycling0.values[s] == "DUE_REVIEW" and review_day[s] < D:
            status[s, review_day[s] * 24:] = "COMPLIANT"
    status = np.where(soh_rep < 76, "DUE_REVIEW", status)
    status = np.where(replaced, "COMPLIANT", status)

    # ------------------------------------------------------------------ energy counter (daily running total)
    energy_cum = np.cumsum(load.reshape(S, D, 24), axis=2).reshape(S, T)

    # ------------------------------------------------------------------ assemble long table
    def flat(a):
        return np.asarray(a).reshape(-1)  # site-major: site 1 all hours, site 2 ...

    df = pd.DataFrame({
        "timestamp": np.tile(ts.values, S),
        "site_id": np.repeat(sites, T),
        "data_volume_gb": flat(vol).round(3),
        "prb_utilization_pct": flat(prb).round(2),
        "active_rrc_connected_users": flat(users).round(1),
        "busy_hour_traffic_ratio": flat(busy).round(4),
        "site_total_power_kw": flat(load).round(4),
        "site_total_energy_kwh": flat(energy_cum).round(4),
        "telecom_dc_load_kw": flat(dc).round(4),
        "site_pue": flat(pue).round(4),
        "grid_voltage_v": flat(volt).round(2),
        "power_source_active": flat(SRC),
        "grid_availability_hours_daily": flat(avail).round(3),
        "battery_soc_pct": flat(SOC_rep).round(2),
        "battery_soh_pct": flat(soh_rep).round(3),
        "solar_generation_kw": flat(solar).round(4),
        "generator_fuel_level_pct": flat(FUEL).round(2),
        "generator_run_hours": flat(GENRUN).astype("int64"),
        "cabinet_internal_temp_c": flat(cab).round(3),
        "battery_bank_temp_c": flat(bank).round(3),
        "ambient_outdoor_temp_c": flat(amb).round(3),
        "hvac_operational_status": flat(hvac_on).astype("int64"),
        "cooling_fan_speed_rpm": flat(rpm).astype("int64"),
        "failed_login_attempts_count": flat(logins).astype("int64"),
        "network_traffic_anomaly_flag": flat(anomaly).astype("int64"),
        "physical_tamper_sensor_status": flat(tamper).astype("int64"),
        "unauthorized_ip_scan_count": flat(scans).astype("int64"),
        "bms_modbus_comm_status": np.where(flat(bms_fault), "FAULT", "NORMAL"),
        "diesel_consumed_liters": flat(DIESEL).round(4),
        "scope_1_carbon_kg_co2e": flat(scope1).round(4),
        "scope_2_carbon_kg_co2e": flat(scope2).round(4),
        "renewable_energy_share_pct": flat(renew).round(3),
        "battery_recycling_status": flat(status),
        "call_setup_success_rate_pct": flat(setup).round(3),
        "call_drop_rate_pct": flat(drop).round(4),
        "avg_dl_throughput_mbps": flat(dl).round(3),
        "avg_ul_throughput_mbps": flat(ul).round(3),
        "latency_rtt_ms": flat(latency).round(3),
        "rsrp_dbm": flat(rsrp).round(3),
        "sinr_db": flat(sinr).round(3),
    })
    for col in STATIC:
        df[col] = np.repeat(sm[col].values, T)
    df = df[COLS]

    # ------------------------------------------------------------------ site master (asset register) + events
    master = sm[STATIC].copy()
    master["region_id"] = region
    master["battery_capacity_kwh_nominal"] = cap_kwh.round(1)
    master["pv_capacity_kwp"] = kwp.round(2)
    master["fuel_tank_l"] = tank_l.round(0)
    master["genset_start_soc_pct"] = gen_thr.round(0)
    master["battery_soh_pct_start"] = sm.battery_soh_pct0.round(2)
    master.reset_index().to_csv(out / "site_master.csv", index=False)
    pd.DataFrame(events).sort_values("start").to_csv(out / "events_ground_truth.csv", index=False)
    df.to_parquet(out / "telecom_synthetic_60d_500sites.parquet", index=False, compression="zstd")
    print(f"rows={len(df):,} cols={df.shape[1]} sites={S} days={D} events={len(events)}")


if __name__ == "__main__":
    main()
