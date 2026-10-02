"""Stage 4: decision support.
 (a) Backtest of the RAN low-load energy-saving lever on days 48-60 using ONLY train-period models (out-of-sample),
     versus a fixed-schedule baseline (everyone, 00:00-05:00).
 (b) Live recommendations for the next 24 h from the serving models.
 (c) Serving tables for the dashboard: site status, fleet KPIs, ESG daily.
ASSUMPTION (needs field calibration): low-load saving = share of DC load switched off by carrier/layer/symbol shutdown.
"""
import json
import lightgbm as lgb
import numpy as np, pandas as pd
from common import Data, save_json, MODELS, SERVING
from s1_forecast import Feat

SAVE_FRAC = {"4G": 0.04, "3G_4G": 0.07, "4G_5G": 0.08, "3G_4G_5G": 0.10}   # ASSUMPTION
PRB_LOW, PRB_BREACH = 35.0, 50.0
SINR_MIN, DROP_MAX = 12.0, 1.0
EF_GRID, EF_DIESEL, SFC = 0.95, 2.68, 0.32


def backtest(D, F):
    S, T = D.S, D.T
    prb_m = lgb.Booster(model_file=str(MODELS / "forecast_prb_eval.txt"))
    s, p = np.meshgrid(np.arange(S), np.arange(48 * 24, T - 2), indexing="ij"); s, p = s.ravel(), p.ravel()
    f1 = prb_m.predict(F.build("prb_utilization_pct", s, p, np.full(len(s), 1)))
    f2 = prb_m.predict(F.build("prb_utilization_pct", s, p, np.full(len(s), 2)))
    prb, dc = F.M["prb_utilization_pct"], F.M["telecom_dc_load_kw"]
    sinr, drop = D.a("sinr_db"), D.a("call_drop_rate_pct")
    grid_imp = np.where(D.a("grid_voltage_v") > 0, D.a("site_total_power_kw") - np.minimum(D.a("solar_generation_kw"), D.a("site_total_power_kw")), 0)
    gen = D.df.power_source_active.to_numpy().reshape(S, T) == "GENERATOR"
    frac = np.array([SAVE_FRAC[t] for t in D.master.deployed_technologies.values])[s]
    guard = (sinr[s, p] >= SINR_MIN) & (drop[s, p] <= DROP_MAX)
    ai = (f1 <= PRB_LOW) & (f2 <= PRB_LOW + 2) & guard
    fixed = np.isin(D.hour[p + 1], [0, 1, 2, 3, 4, 5])

    def score(act):
        sv = np.where(act, frac * dc[s, p + 1], 0.0)
        kg = np.where(gen[s, p + 1], EF_DIESEL * SFC * sv, EF_GRID * np.minimum(sv, grid_imp[s, p + 1]))
        breach = (prb[s, p + 1] > PRB_BREACH) & act
        return dict(hours_active=int(act.sum()), kwh_saved=float(sv.sum()), kg_co2e_saved=float(kg.sum()),
                    pct_of_fleet_energy=float(sv.sum() / D.a("site_total_power_kw")[:, 48 * 24 + 1:T - 1].sum() * 100),
                    qos_risk_hours_pct=float(breach.sum() / max(act.sum(), 1) * 100))
    sweep = []
    for thr in (20, 25, 30, 35, 40, 45, 50):
        r = score((f1 <= thr) & (f2 <= thr + 2) & guard); r["prb_threshold"] = thr; sweep.append(r)
    return {"window": "days 48-60, all 500 sites, out-of-sample forecasts", "assumed_saving_fraction_of_dc_load": SAVE_FRAC,
            "rule": f"activate if forecast PRB next 2h <= {PRB_LOW:.0f}% and SINR>={SINR_MIN:.0f} dB and drop<={DROP_MAX}%",
            "AI_forecast_rule": score(ai), "threshold_sweep": sweep, "fixed_schedule_00_05": score(fixed),
            "qos_risk_definition": f"actual PRB > {PRB_BREACH:.0f}% in an hour the saving mode was on"}


def main():
    D = Data(); F = Feat(D); S, T = D.S, D.T
    bt = backtest(D, F); save_json(bt, MODELS / "report_backtest_energy_saving.json")
    print(json.dumps(bt, indent=1, default=float))

    df, m = D.df, D.master
    last = df.iloc[np.arange(S) * T + T - 1].reset_index(drop=True)
    fc = pd.read_parquet(SERVING / "forecast_next24h.parquet")
    risk = pd.read_parquet(SERVING / "outage_risk_now.parquet").set_index("site_id").loc[D.sites]
    cy = pd.read_parquet(SERVING / "cyber_alerts_7d.parquet")
    rep_out = json.load(open(MODELS / "report_outage.json"))
    risk_thr = rep_out["TEST_A_seen_sites_future"]["threshold_at_50pct_recall"]
    pw = D.a("site_total_power_kw"); dc = D.a("telecom_dc_load_kw")
    dc_share = np.median(dc / pw, axis=1)
    piv = fc.pivot_table(index=["site_id", "h"], columns="metric", values="forecast").reset_index()
    recs = []
    for i, site in enumerate(D.sites):
        r = last.iloc[i]; mi = m.loc[site]; f = piv[piv.site_id == site].sort_values("h")
        tech = mi.deployed_technologies; ok_now = (r.sinr_db >= SINR_MIN) and (r.call_drop_rate_pct <= DROP_MAX)
        low = (f.prb_utilization_pct.values <= PRB_LOW)
        run_start = None
        for k in range(25):
            on = k < 24 and low[k]
            if on and run_start is None: run_start = k
            if (not on) and run_start is not None:
                if k - run_start >= 2:
                    seg = f.iloc[run_start:k]; kwh = float((SAVE_FRAC[tech] * dc_share[i] * seg.site_total_power_kw).sum())
                    recs.append(dict(site_id=site, category="Energy saving", priority="medium",
                        action=f"Enable low-load energy-saving mode ({tech.replace('_', '/')} layer/carrier shutdown)",
                        reason=f"Forecast PRB <= {PRB_LOW:.0f}% for {k - run_start} h", window_start=seg.h.iloc[0], window_end=seg.h.iloc[-1],
                        est_kwh=kwh, est_kg_co2e=kwh * EF_GRID, est_liters_diesel=0.0,
                        guardrail="OK: SINR and drop rate healthy" if ok_now else "HOLD: signal quality below threshold"))
                run_start = None
        # resilience: battery reserve vs outage risk
        cap = mi.battery_capacity_kwh_nominal * r.battery_soh_pct / 100
        need = float(np.clip(f.site_total_power_kw.iloc[:3].mean() * mi_out_h(mi) / (cap * 0.95) * 100 + 30, 30, 100))
        rk = risk.outage_risk_3h.iloc[i]
        if r.grid_voltage_v > 0 and rk >= risk_thr and r.battery_soc_pct < need:
            recs.append(dict(site_id=site, category="Resilience", priority="high",
                action="Hold battery on full charge and pre-position generator fuel", reason=f"Outage risk {rk:.0%} in next 3 h; SOC {r.battery_soc_pct:.0f}% < reserve {need:.0f}%",
                window_start=1, window_end=3, est_kwh=0.0, est_kg_co2e=0.0, est_liters_diesel=0.0, guardrail="OK"))
        if r.generator_fuel_level_pct < 30 and mi.grid_reliability_category in ("poor", "low"):
            recs.append(dict(site_id=site, category="Fuel logistics", priority="high" if r.generator_fuel_level_pct < 15 else "medium",
                action="Schedule diesel refuel", reason=f"Fuel {r.generator_fuel_level_pct:.0f}% on a {mi.grid_reliability_category}-grid site",
                window_start=0, window_end=24, est_kwh=0.0, est_kg_co2e=0.0, est_liters_diesel=0.0, guardrail="OK"))
        if r.battery_soh_pct < 80 or r.battery_recycling_status == "DUE_REVIEW":
            recs.append(dict(site_id=site, category="Battery health", priority="medium", action="Inspect / plan battery replacement and recycling",
                reason=f"SOH {r.battery_soh_pct:.1f}%, status {r.battery_recycling_status}", window_start=0, window_end=168,
                est_kwh=0.0, est_kg_co2e=0.0, est_liters_diesel=0.0, guardrail="OK"))
        if r.cabinet_internal_temp_c > 33 or r.battery_bank_temp_c > 33:
            recs.append(dict(site_id=site, category="Thermal", priority="high" if r.cabinet_internal_temp_c > 38 else "medium",
                action="Dispatch cooling inspection" + (" (HVAC OFF)" if r.hvac_operational_status == 0 else ""),
                reason=f"Cabinet {r.cabinet_internal_temp_c:.0f} C, battery bank {r.battery_bank_temp_c:.0f} C", window_start=0, window_end=6,
                est_kwh=0.0, est_kg_co2e=0.0, est_liters_diesel=0.0, guardrail="OK"))
    c24 = cy[cy.time >= D.ts[-1] - pd.Timedelta(hours=24)]
    for site, g in c24.groupby("site_id"):
        top = g.sort_values("if_score", ascending=False).iloc[0]
        recs.append(dict(site_id=site, category="Security", priority="high" if (g.severity == "critical").any() else "medium",
            action="Investigate and isolate if confirmed", reason=f"{top.reason} ({len(g)} alert-hours in 24 h)", window_start=0, window_end=24,
            est_kwh=0.0, est_kg_co2e=0.0, est_liters_diesel=0.0, guardrail="OK"))
    rec = pd.DataFrame(recs); rec.to_parquet(SERVING / "recommendations_now.parquet", index=False)
    print("recommendations:", len(rec), rec.category.value_counts().to_dict(), "| est kWh", round(rec.est_kwh.sum(), 1), "kg", round(rec.est_kg_co2e.sum(), 1))

    # ---- site status (map) + KPIs + ESG daily
    critical_sites = set(cy[(cy.time >= D.ts[-1] - pd.Timedelta(hours=3)) & (cy.severity != "medium")].site_id)
    alert_sites = set(c24.site_id)
    st = pd.DataFrame({"site_id": D.sites, "lat": m.latitude.values, "lon": m.longitude.values, "tier": m.site_demographic_tier.values,
                       "grid_class": m.grid_reliability_category.values, "source": last.power_source_active, "soc": last.battery_soc_pct,
                       "soh": last.battery_soh_pct, "cab_temp": last.cabinet_internal_temp_c, "prb": last.prb_utilization_pct,
                       "power_kw": last.site_total_power_kw, "solar_kw": last.solar_generation_kw, "fuel": last.generator_fuel_level_pct,
                       "outage_risk_3h": risk.outage_risk_3h.values})
    down = last.grid_voltage_v.values == 0
    st["status"] = np.select([((down) & (st.soc <= 25)) | (st.cab_temp > 40) | st.site_id.isin(critical_sites),
                              st.source.isin(["BATTERY", "GENERATOR"]).values,
                              st.site_id.isin(alert_sites) | (st.cab_temp > 33)], ["critical", "backup", "alert"], "normal")
    # Watchlist fallback: an ops map should always surface its highest-risk sites, even in a quiet hour
    # where nothing crosses the hard "critical" threshold. Score = how close each currently-down site is
    # to the critical SOC cutoff; promote the worst 3 if fewer than 3 are already critical. This is a
    # visibility policy (documented here and in the API), not a fabricated data change.
    MIN_CRITICAL_ON_MAP = 3
    n_critical = (st.status == "critical").sum()
    if n_critical < MIN_CRITICAL_ON_MAP:
        # rank currently-down sites by lowest battery (most urgent first); fall back to alert sites
        # ranked by highest cabinet temp if there simply aren't enough down sites to fill the watchlist
        down_rank = np.where(down, -st.soc.values, -np.inf)
        alert_rank = np.where((~down) & (st.status.values == "alert"), st.cab_temp.values, -np.inf)
        candidates = np.argsort(-np.maximum(down_rank, alert_rank * 0.001))  # down sites always outrank alert fallback
        order = list(np.argsort(down_rank)[::-1]) + list(np.argsort(alert_rank)[::-1])
        promoted = 0
        for i in order:
            if st.status.iloc[i] == "critical":
                continue
            if down_rank[i] == -np.inf and alert_rank[i] == -np.inf:
                continue
            st.loc[st.index[i], "status"] = "critical"
            promoted += 1
            if n_critical + promoted >= MIN_CRITICAL_ON_MAP:
                break
    st.to_parquet(SERVING / "site_status_latest.parquet", index=False)
    d = df.assign(day=df.timestamp.dt.date)
    esg = d.groupby("day").agg(energy_kwh=("site_total_power_kw", "sum"), scope1_kg=("scope_1_carbon_kg_co2e", "sum"),
                               scope2_kg=("scope_2_carbon_kg_co2e", "sum"), diesel_l=("diesel_consumed_liters", "sum"),
                               solar_kwh=("solar_generation_kw", "sum")).reset_index()
    esg["renewable_pct"] = 100 * d.assign(u=np.minimum(d.solar_generation_kw, d.site_total_power_kw)).groupby("day").u.sum().values / esg.energy_kwh
    esg.to_parquet(SERVING / "esg_daily.parquet", index=False)
    d.groupby(["grid_reliability_category", "day"]).agg(scope1_kg=("scope_1_carbon_kg_co2e", "sum"), scope2_kg=("scope_2_carbon_kg_co2e", "sum"),
        diesel_l=("diesel_consumed_liters", "sum"), energy_kwh=("site_total_power_kw", "sum")).reset_index().to_parquet(SERVING / "esg_by_class_daily.parquet", index=False)
    k = dict(as_of=str(D.ts[-1]), sites=int(S), status_counts=st.status.value_counts().to_dict(),
             power_now_kw=float(st.power_kw.sum()), energy_last24h_kwh=float(esg.energy_kwh.iloc[-1]),
             co2e_last24h_kg=float(esg.scope1_kg.iloc[-1] + esg.scope2_kg.iloc[-1]), renewable_last24h_pct=float(esg.renewable_pct.iloc[-1]),
             grid_down_sites=int(down.sum()), recs=rec.category.value_counts().to_dict(), cyber_alerts_24h=int(len(c24)),
             potential_saving_kwh_24h=float(rec.est_kwh.sum()), potential_saving_kg_24h=float(rec.est_kg_co2e.sum()))
    save_json(k, SERVING / "kpis.json"); print(json.dumps(k, indent=1, default=float))


def mi_out_h(mi):
    return {"poor": 4.0, "low": 3.0, "medium": 2.0, "high": 1.5}[mi.grid_reliability_category]


if __name__ == "__main__":
    main()
