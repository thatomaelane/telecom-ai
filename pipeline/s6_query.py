"""Stage 6: the query the dashboard actually needs.
   Pick a site + a date/time -> current status, next-24h predictions (network, energy, cost, CO2, ESG,
   cyber), and one smart decision with the reasoning and numbers behind it.

This is deliberately a plain Python module (no web framework) so it can be tested from the terminal
before it is wrapped in FastAPI. `snapshot()` is the one function the API will call.
"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb, joblib
from common import Data, MODELS, SERVING, read_parquet_lowmem
from s1_forecast import Feat
from s5_orchestrator import CFG
from s3_cyber import build as build_cyber_features, FEATS as CYBER_FEATS

PRB_LOW, SINR_MIN, DROP_MAX = 35.0, 12.0, 1.0
ARPU_R_PER_MONTH = 120.0   # STUDIED ESTIMATE: recent Vodacom/MTN SA blended ARPU disclosures cluster ~R90-130/month
                           # (Vodacom SA prepaid ~R58-60, contract ~R290-390, blended trending up with data growth).
                           # No single current published blended figure found - confirm against latest investor results before final submission.
REVENUE_PER_USER_HOUR = ARPU_R_PER_MONTH / 30 / 24
GEN_SERVICE_INTERVAL_H = 250.0     # ASSUMPTION: typical diesel genset minor-service interval
BATTERY_SERVICE_SOH_PCT = 75.0     # ASSUMPTION: SOH threshold at which a telecom battery bank is typically scheduled for service/replacement
THERMAL_STRESS_C = 38.0


class Engine:
    """Loads every model + the site/hour matrices ONCE. Reuse one Engine instance across queries
    (e.g. as a FastAPI app-level singleton) - do not rebuild it per request, it takes ~10s to load."""

    def __init__(self):
        self.D = Data(); self.F = Feat(self.D); S, T = self.D.S, self.D.T
        self.site_idx = {s: i for i, s in enumerate(self.D.sites)}
        self.fc_models = {short: lgb.Booster(model_file=str(MODELS / f"forecast_{short}.txt"))
                           for short in ["vol", "prb", "power"]}
        self.risk_model = lgb.Booster(model_file=str(MODELS / "outage_risk.txt"))
        cy = joblib.load(MODELS / "cyber_isoforest.joblib")
        self.iso, self.thr_rule, self.thr_if = cy["model"], cy["thr_rule_z"], cy["thr_if"]
        self.Z = build_cyber_features(self.D)
        self.hourly = read_parquet_lowmem(SERVING / "orchestrator_hourly.parquet")
        self.hourly["timestamp"] = pd.to_datetime(self.hourly.timestamp)
        self.sites_tbl = pd.read_parquet(SERVING / "orchestrator_sites.parquet").set_index("site_id")
        self.ts_index = pd.Series(np.arange(T), index=self.D.ts[:T])
        self._precompute_maintenance()
        self._precompute_peer_groups()

    # ---------------------------------------------------------------- maintenance prediction (vectorised, once)
    def _precompute_maintenance(self):
        D = self.D; S, T = D.S, D.T; days = T / 24
        soh = D.a("battery_soh_pct"); gen_h = D.a("generator_run_hours"); cab = D.a("cabinet_internal_temp_c")
        x = np.arange(T) / 24.0
        xm = x.mean(); slope = ((x - xm) * (soh - soh.mean(axis=1, keepdims=True))).sum(axis=1) / ((x - xm) ** 2).sum()
        soh_now = soh[:, -1]
        days_to_soh_threshold = np.where(slope < -1e-4, (soh_now - BATTERY_SERVICE_SOH_PCT) / -slope, 9999.0)
        days_to_soh_threshold = np.clip(days_to_soh_threshold, 0, 9999)
        gen_h_rate = gen_h.sum(axis=1) / days           # generator hours accumulated per day
        gen_h_total = gen_h.sum(axis=1)
        hours_since_service = gen_h_total % GEN_SERVICE_INTERVAL_H
        days_to_gen_service = np.where(gen_h_rate > 1e-3, (GEN_SERVICE_INTERVAL_H - hours_since_service) / gen_h_rate, 9999.0)
        thermal_events = (cab > THERMAL_STRESS_C).sum(axis=1)
        self.maintenance = pd.DataFrame({
            "site_id": D.sites, "battery_soh_now": soh_now.round(1), "battery_soh_trend_per_day": slope.round(4),
            "days_to_battery_service": np.round(days_to_soh_threshold, 0),
            "generator_hours_total": gen_h_total.round(0), "days_to_generator_service": np.round(days_to_gen_service, 0),
            "thermal_stress_events": thermal_events,
        }).set_index("site_id")

    def maintenance_for(self, site_id):
        r = self.maintenance.loc[site_id]
        items = []
        if r.days_to_battery_service < 45:
            items.append(dict(component="Battery bank", urgency="high" if r.days_to_battery_service < 14 else "medium",
                eta_days=int(r.days_to_battery_service),
                reason=f"SOH declining {abs(r.battery_soh_trend_per_day)*30:.1f} pts/month, now {r.battery_soh_now:.0f}% - will cross the {BATTERY_SERVICE_SOH_PCT:.0f}% service threshold in ~{int(r.days_to_battery_service)} days",
                est_avoided_cost_zar=8000.0))    # ASSUMPTION: emergency callout + expedited replacement premium avoided by planned service
        if r.days_to_generator_service < 30:
            items.append(dict(component="Generator", urgency="high" if r.days_to_generator_service < 7 else "medium",
                eta_days=int(r.days_to_generator_service),
                reason=f"{r.generator_hours_total:.0f} running hours accumulated - due for scheduled service in ~{int(r.days_to_generator_service)} days",
                est_avoided_cost_zar=3500.0))
        if r.thermal_stress_events > 3:
            items.append(dict(component="Cooling / HVAC", urgency="medium", eta_days=None,
                reason=f"Cabinet exceeded {THERMAL_STRESS_C:.0f} C on {int(r.thermal_stress_events)} occasions in 60 days - elevated equipment stress and failure risk",
                est_avoided_cost_zar=12000.0))   # ASSUMPTION: cost of an unplanned outage from a cooling-related equipment failure
        return items

    # ---------------------------------------------------------------- peer comparison (for "X% more energy than comparable sites")
    def _precompute_peer_groups(self):
        D = self.D
        self._peer_tier = D.static.tier.values.copy()
        self._peer_tech = D.static.tech.values.copy()

    def peer_energy_comparison(self, i, p):
        D = self.D
        peers = np.where((self._peer_tier == self._peer_tier[i]) & (self._peer_tech == self._peer_tech[i]))[0]
        peers = peers[peers != i]
        if len(peers) < 5:
            return None
        pw = D.a("site_total_power_kw"); vol = D.a("data_volume_gb")
        my_pw, my_vol = pw[i, p], vol[i, p]
        vb = np.abs(vol[peers, p] - my_vol) <= max(my_vol * 0.35, 1.0)
        sel = peers[vb] if vb.sum() >= 5 else peers
        peer_mean = pw[sel, p].mean()
        if peer_mean <= 0:
            return None
        pct = (my_pw - peer_mean) / peer_mean * 100
        return dict(peer_mean_kw=round(float(peer_mean), 2), site_kw=round(float(my_pw), 2), pct_vs_peers=round(float(pct), 1), n_peers=int(len(sel)))

    # ---------------------------------------------------------------- helpers
    def _p(self, site_id, when):
        i = self.site_idx[site_id]
        t = pd.Timestamp(when).floor("h")
        if t in self.ts_index.index:
            p = int(self.ts_index[t])
        else:
            p = int(self.ts_index.iloc[np.abs((self.ts_index.index - t).total_seconds()).argmin()])
        return i, p

    def _forecast(self, tgt, i, p, horizons):
        s = np.full(len(horizons), i); pp = np.full(len(horizons), p); hh = np.array(horizons)
        X = self.F.build(tgt, s, pp, hh)
        return self.fc_models[{"data_volume_gb": "vol", "prb_utilization_pct": "prb", "site_total_power_kw": "power"}[tgt]].predict(X)

    def _outage_risk(self, i, p):
        if self.D.a("grid_voltage_v")[i, p] == 0:
            return None  # already down
        volt = self.D.a("grid_voltage_v"); avail = self.D.a("grid_availability_hours_daily")
        down = volt == 0
        onehot = np.zeros((self.D.region.max() + 1, self.D.S), np.float32); onehot[self.D.region, np.arange(self.D.S)] = 1
        reg_frac = (onehot @ down.astype(np.float32)) / onehot.sum(1, keepdims=True)
        cnt = 999
        for t in range(max(0, p - 500), p + 1):
            cnt = 0 if down[i, t] else cnt + 1
        X = pd.DataFrame([{"volt": volt[i, p], "volt_lag1": volt[i, max(p - 1, 0)],
                           "volt_min3": min(volt[i, max(p - k, 0)] for k in range(3) if volt[i, max(p - k, 0)] > 0),
                           "sag": 230 - volt[i, p], "avail24": avail[i, p], "since_outage_h": min(cnt, 999),
                           "region_frac_down": reg_frac[self.D.region[i], p], "hour": self.D.hour[p], "dow": self.D.dow[p],
                           "cat": self.D.static.cat.values[i], "lat": self.D.static.lat.values[i], "lon": self.D.static.lon.values[i],
                           "backhaul": self.D.static.backhaul.values[i], "tier": self.D.static.tier.values[i]}])
        return float(self.risk_model.predict(X)[0])

    def _cyber(self, i, p):
        x = np.array([[self.Z[f][i, p] for f in CYBER_FEATS]])
        score = float(-self.iso.score_samples(x)[0])
        rule = float(max(self.Z["z_logins"][i, p], self.Z["z_scans"][i, p]))
        phys = bool((self.Z["fuel_drop"][i, p] > 10) or (self.Z["tamper"][i, p] > 0 and self.Z["soc_drop"][i, p] > 20))
        alert = (rule >= self.thr_rule) or (score >= self.thr_if) or phys
        if not alert:
            reason = "No anomaly detected"
        elif phys:
            reason = "Possible fuel/battery theft (tamper sensor + sudden drop)"
        elif self.Z["z_vol"][i, p] > 4 or self.Z["z_users"][i, p] > 4:
            reason = "Traffic flood / DDoS-like surge"
        elif self.Z["z_scans"][i, p] > 4:
            reason = "Port-scan / reconnaissance burst"
        elif self.Z["z_logins"][i, p] > 4:
            reason = "Brute-force login attempts"
        elif self.Z["bms_fault"][i, p]:
            reason = "BMS/Modbus communication fault"
        else:
            reason = "Combined behavioural anomaly"
        return dict(alert=bool(alert), isolation_forest_score=round(score, 3), login_scan_zscore=round(rule, 2), reason=reason)

    def _opex(self, row, cyber_alert, call_setup, call_drop, cycles_frac):
        c = CFG
        c_grid = float(row.smart_grid_kwh * c["TARIFF_R_PER_KWH"])
        c_diesel = float(row.smart_diesel_l * c["DIESEL_R_PER_L"])
        c_batt = float(cycles_frac * c["BATT_WEAR_R_PER_CYCLE"])
        c_carbon = float(row.smart_co2e_kg * c["CARBON_SHADOW_R_PER_KG"])
        c_cyber = float(c["CYBER_INCIDENT_R"]) if cyber_alert else 0.0
        c_sla = float((100 - call_setup) * c["SLA_R_PER_DROPPED_PCT"]) if (call_drop > DROP_MAX or call_setup < 95.0) else 0.0
        total = c_grid + c_diesel + c_batt + c_carbon + c_cyber + c_sla
        return dict(grid_zar=round(c_grid, 2), diesel_zar=round(c_diesel, 2), battery_wear_zar=round(c_batt, 2),
                    carbon_shadow_zar=round(c_carbon, 2), cyber_risk_zar=round(c_cyber, 2), sla_penalty_zar=round(c_sla, 2),
                    total_zar=round(total, 2))

    def financial_impact(self, row, hrow, active_users, cyber_alert, call_setup, call_drop, grid_up, cycles_frac, maintenance_items):
        """Replaces the old generic OPEX breakdown with components that map to real telco economics:
        energy cost (sourced ZAR/kWh + ZAR/L), an ESG shadow carbon price (sourced R308/t headline), revenue-at-risk
        (ARPU-based: what the site's subscriber base is worth per hour, discounted by how degraded service is right
        now), and incident cost (cyber response + avoidable maintenance callouts)."""
        c = CFG
        energy_zar = float(hrow.smart_grid_kwh * c["TARIFF_R_PER_KWH"] + hrow.smart_diesel_l * c["DIESEL_R_PER_L"])
        battery_wear_zar = float(cycles_frac * c["BATT_WEAR_R_PER_CYCLE"])
        carbon_zar = float(hrow.smart_co2e_kg * c["CARBON_SHADOW_R_PER_KG"])
        # service degradation factor: 1.0 = site fully down (all revenue-generating traffic lost this hour),
        # scaled down for a live-but-degraded site using the shortfall in setup success / excess drop rate.
        if not grid_up and hrow.smart_gen_on == 0 and hrow.smart_soc <= 10:
            degradation = 1.0
        else:
            degradation = float(np.clip((100 - call_setup) / 5 + max(call_drop - DROP_MAX, 0) / 5, 0, 1))
        revenue_at_risk_zar = float(active_users * REVENUE_PER_USER_HOUR * degradation)
        cyber_zar = float(c["CYBER_INCIDENT_R"]) if cyber_alert else 0.0
        maint_zar = float(sum(m["est_avoided_cost_zar"] for m in maintenance_items if m["urgency"] == "high")) / (30 * 24)  # amortised per hour
        total = energy_zar + battery_wear_zar + carbon_zar + revenue_at_risk_zar + cyber_zar + maint_zar
        return dict(energy_zar=round(energy_zar, 2), battery_wear_zar=round(battery_wear_zar, 2), carbon_shadow_zar=round(carbon_zar, 2),
                    revenue_at_risk_zar=round(revenue_at_risk_zar, 2), incident_zar=round(cyber_zar + maint_zar, 2),
                    total_zar=round(total, 2), degradation_factor=round(degradation, 2),
                    arpu_note=f"Revenue at risk uses an estimated blended ARPU of R{ARPU_R_PER_MONTH:.0f}/month (R{REVENUE_PER_USER_HOUR:.3f}/user/hour) - confirm against the latest published operator ARPU before final submission.")

    # ---------------------------------------------------------------- main entry point
    def snapshot(self, site_id, when, horizon_h=24):
        D = self.D
        i, p = self._p(site_id, when)
        row = D.df.iloc[i * D.T + p]
        mi = D.master.loc[site_id]
        site_rows = self.hourly[self.hourly.site_id == site_id].reset_index(drop=True)
        hrow = site_rows.iloc[p]
        soc_prev = site_rows.iloc[max(p - 1, 0)].smart_soc
        cycles = abs(float(hrow.smart_soc) - float(soc_prev)) / 100.0
        cyber = self._cyber(i, p)
        maint_items = self.maintenance_for(site_id)
        fin = self.financial_impact(row, hrow, float(row.active_rrc_connected_users), cyber["alert"],
                                     row.call_setup_success_rate_pct, row.call_drop_rate_pct, row.grid_voltage_v > 0, cycles, maint_items)
        peer = self.peer_energy_comparison(i, p)

        horizons = list(range(1, horizon_h + 1))
        vol_f = self._forecast("data_volume_gb", i, p, horizons)
        prb_f = self._forecast("prb_utilization_pct", i, p, horizons)
        pow_f = self._forecast("site_total_power_kw", i, p, horizons)
        future_ts = [str(D.ts_ext[p + h]) for h in horizons]
        outage_risk = self._outage_risk(i, p)

        # -- smart decision (same guardrails as the fleet recommendation engine, applied to this one query)
        ok = (row.sinr_db >= SINR_MIN) and (row.call_drop_rate_pct <= DROP_MAX)
        low_run = int(np.sum(prb_f[:6] <= PRB_LOW))
        decisions = []
        if low_run >= 2 and ok:
            frac = CFG["SAVE_FRAC"][mi.deployed_technologies]
            kwh = float(frac * pow_f[:low_run].sum())
            decisions.append(dict(category="Energy saving", priority="medium",
                action=f"Enable low-load energy-saving mode for the next {low_run} h ({mi.deployed_technologies.replace('_','/')} layer shutdown)",
                reason=f"Forecast PRB <= {PRB_LOW:.0f}% and signal quality healthy (SINR {row.sinr_db:.1f} dB, drop {row.call_drop_rate_pct:.2f}%)",
                est_kwh=round(kwh, 1), est_kg_co2e=round(kwh * CFG["EF_GRID"], 1), est_zar=round(kwh * CFG["TARIFF_R_PER_KWH"], 0)))
        elif low_run >= 2 and not ok:
            decisions.append(dict(category="Energy saving", priority="low", action="Hold off on energy saving",
                reason=f"Low load forecast, but signal quality already weak (SINR {row.sinr_db:.1f} dB / drop {row.call_drop_rate_pct:.2f}%) - saving now risks availability",
                est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if outage_risk is not None and outage_risk >= 0.13 and row.battery_soc_pct < 60:
            decisions.append(dict(category="Resilience", priority="high",
                action="Hold battery charge and pre-position generator fuel",
                reason=f"Outage risk {outage_risk:.0%} in next 3 h, SOC only {row.battery_soc_pct:.0f}%", est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if row.generator_fuel_level_pct < 30 and mi.grid_reliability_category in ("poor", "low"):
            decisions.append(dict(category="Fuel logistics", priority="high" if row.generator_fuel_level_pct < 15 else "medium",
                action="Schedule diesel refuel", reason=f"Fuel at {row.generator_fuel_level_pct:.0f}% on a {mi.grid_reliability_category}-grid site", est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if peer is not None and peer["pct_vs_peers"] > 20:
            decisions.append(dict(category="Energy", priority="medium",
                action=f"Investigate cooling equipment and non-critical component draw at {site_id}",
                reason=f"Consuming {peer['pct_vs_peers']:.0f}% more power than {peer['n_peers']} comparable sites (same tier/technology, similar traffic) - {peer['site_kw']:.1f} kW vs a {peer['peer_mean_kw']:.1f} kW peer average",
                potential_issue="Equipment inefficiency or abnormal power draw (e.g. a cooling fault or a misconfigured radio unit running at full power outside busy hours)",
                est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if cyber["alert"]:
            decisions.append(dict(category="Security", priority="high", action="Isolate the affected equipment and start an incident investigation",
                reason=cyber["reason"], potential_issue="Compromised network equipment or abnormal IoT/OT activity - possible unauthorised access attempt",
                est_kwh=0.0, est_kg_co2e=0.0, est_zar=CFG["CYBER_INCIDENT_R"]))
        if row.cabinet_internal_temp_c > 33:
            decisions.append(dict(category="Thermal", priority="high" if row.cabinet_internal_temp_c > 38 else "medium",
                action="Dispatch cooling inspection", reason=f"Cabinet at {row.cabinet_internal_temp_c:.0f} C", est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        for m in maint_items:
            decisions.append(dict(category="Maintenance", priority=m["urgency"], action=f"Schedule {m['component'].lower()} service (in ~{m['eta_days']} days)" if m["eta_days"] else f"Inspect {m['component'].lower()}",
                reason=m["reason"], potential_issue=None, est_kwh=0.0, est_kg_co2e=0.0, est_zar=-m["est_avoided_cost_zar"]))
        if not decisions:
            decisions.append(dict(category="None", priority="low", action="No action needed", reason="All indicators within normal range", potential_issue=None, est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        for d in decisions:
            d.setdefault("potential_issue", None)

        # -- prediction summary band (network availability, trends, monthly saving estimate)
        w = min(p + 1, 7 * 24)
        avail_pct = float(100 * (1 - site_rows.iloc[max(0, p - w + 1):p + 1].smart_risk.mean()))
        w24 = min(p + 1, 24); wprev = min(max(0, p - w24 + 1), 6 * 24)
        power_today = site_rows.iloc[max(0, p - w24 + 1):p + 1].load_kw.mean()
        power_prev = site_rows.iloc[max(0, p - w24 + 1 - wprev):max(0, p - w24 + 1)].load_kw.mean() if wprev > 0 else np.nan
        co2_today = (site_rows.iloc[max(0, p - w24 + 1):p + 1].smart_co2e_kg).sum()
        co2_prev = (site_rows.iloc[max(0, p - w24 + 1 - wprev):max(0, p - w24 + 1)].smart_co2e_kg).sum() if wprev > 0 else np.nan
        monthly_saving = sum(d["est_zar"] for d in decisions if d["category"] == "Energy saving") * 30
        summary = {
            "network_availability_pct": round(avail_pct, 1),
            "energy_today_kwh": round(float(power_today * w24), 1),
            "energy_change_pct": round(float((power_today - power_prev) / power_prev * 100), 1) if wprev > 0 and power_prev > 0 else None,
            "co2_today_kg": round(float(co2_today), 1),
            "co2_change_pct": round(float((co2_today - co2_prev) / co2_prev * 100), 1) if wprev > 0 and co2_prev > 0 else None,
            "cyber_alert_now": cyber["alert"],
            "estimated_monthly_saving_zar": round(float(monthly_saving), 0),
        }

        return {
            "site_id": site_id, "as_of": str(D.ts[p]),
            "site_info": {"tier": mi.site_demographic_tier, "grid_class": mi.grid_reliability_category,
                          "backhaul": mi.backhaul_technology, "tech": mi.deployed_technologies,
                          "enclosure": mi.enclosure_type, "lat": float(mi.latitude), "lon": float(mi.longitude)},
            "current": {
                "network_demand": {"data_volume_gb": round(float(row.data_volume_gb), 2), "prb_utilization_pct": round(float(row.prb_utilization_pct), 1),
                                    "active_users": round(float(row.active_rrc_connected_users)), "busy_hour_ratio": round(float(row.busy_hour_traffic_ratio), 2)},
                "energy": {"power_kw": round(float(row.site_total_power_kw), 2), "power_source": row.power_source_active,
                          "battery_soc_pct": round(float(row.battery_soc_pct), 1), "battery_soh_pct": round(float(row.battery_soh_pct), 1),
                          "solar_kw": round(float(row.solar_generation_kw), 2), "fuel_pct": round(float(row.generator_fuel_level_pct), 1),
                          "grid_up": bool(row.grid_voltage_v > 0), "grid_voltage_v": round(float(row.grid_voltage_v), 1)},
                "customer_experience": {"dl_mbps": round(float(row.avg_dl_throughput_mbps), 1), "ul_mbps": round(float(row.avg_ul_throughput_mbps), 1),
                                         "latency_ms": round(float(row.latency_rtt_ms), 1), "call_setup_success_pct": round(float(row.call_setup_success_rate_pct), 2),
                                         "call_drop_pct": round(float(row.call_drop_rate_pct), 2), "sinr_db": round(float(row.sinr_db), 1), "rsrp_dbm": round(float(row.rsrp_dbm), 1)},
                "thermal": {"cabinet_c": round(float(row.cabinet_internal_temp_c), 1), "battery_bank_c": round(float(row.battery_bank_temp_c), 1),
                           "ambient_c": round(float(row.ambient_outdoor_temp_c), 1), "hvac_on": bool(row.hvac_operational_status)},
            },
            "predictions_next_h": {"timestamps": future_ts, "data_volume_gb": [round(float(x), 2) for x in vol_f],
                                    "prb_utilization_pct": [round(float(x), 1) for x in prb_f], "power_kw": [round(float(x), 2) for x in pow_f],
                                    "grid_outage_risk_next_3h": None if outage_risk is None else round(outage_risk, 3)},
            "financial_impact_this_hour_zar": fin,
            "esg_this_hour": {"scope1_kg_co2e": round(float(row.scope_1_carbon_kg_co2e), 3), "scope2_kg_co2e": round(float(row.scope_2_carbon_kg_co2e), 3),
                              "renewable_share_pct": round(float(row.renewable_energy_share_pct), 1), "battery_recycling_status": row.battery_recycling_status},
            "cybersecurity": cyber,
            "maintenance": maint_items,
            "peer_comparison": peer,
            "prediction_summary": summary,
            "smart_decisions": decisions,
        }


    # ---------------------------------------------------------------- date-range history (Site Explorer "Trend" view)
    def history(self, site_id, start, end):
        D = self.D
        i, p0 = self._p(site_id, start); _, p1 = self._p(site_id, end)
        p0, p1 = min(p0, p1), max(p0, p1)
        if p1 - p0 > 24 * 92:
            p1 = p0 + 24 * 92   # cap at ~92 days of hourly points so the response stays a sane size
        site_rows = self.hourly[self.hourly.site_id == site_id].reset_index(drop=True).iloc[p0:p1 + 1]
        df = D.df.iloc[i * D.T + p0: i * D.T + p1 + 1]
        out = pd.DataFrame({
            "timestamp": df.timestamp.values, "data_volume_gb": df.data_volume_gb.values, "prb_utilization_pct": df.prb_utilization_pct.values,
            "active_users": df.active_rrc_connected_users.values, "power_kw": df.site_total_power_kw.values,
            "battery_soc_pct": df.battery_soc_pct.values, "power_source": df.power_source_active.values,
            "call_drop_pct": df.call_drop_rate_pct.values, "call_setup_success_pct": df.call_setup_success_rate_pct.values,
            "scope1_kg": df.scope_1_carbon_kg_co2e.values, "scope2_kg": df.scope_2_carbon_kg_co2e.values,
            "cost_zar": site_rows.smart_cost_zar.values, "risk": site_rows.smart_risk.values,
        })
        n = len(out)
        return {"site_id": site_id, "n_hours": n,
                "summary": {"avg_power_kw": round(float(out.power_kw.mean()), 2), "peak_power_kw": round(float(out.power_kw.max()), 2),
                            "total_co2e_kg": round(float((out.scope1_kg + out.scope2_kg).sum()), 1), "total_cost_zar": round(float(out.cost_zar.sum()), 0),
                            "downtime_hours": int(out.risk.sum()), "avg_call_drop_pct": round(float(out.call_drop_pct.mean()), 2),
                            "avg_setup_success_pct": round(float(out.call_setup_success_pct.mean()), 2)},
                "series": json.loads(out.to_json(orient="records", date_format="iso"))}

    # ---------------------------------------------------------------- instant assessment (Simulate: manual telemetry, no history needed)
    def instant_assessment(self, t):
        """t: dict of manually entered telemetry. Does not use the ML forecast models (no history/lag features
        available for a one-off manual reading) - uses the same guardrail thresholds and cost formulas as
        snapshot(), applied directly to the given readings. Suitable for a field technician taking a live
        reading and needing an immediate decision."""
        c = CFG
        tier, tech, grid_class = t.get("tier", "medium"), t.get("tech", "4G_5G"), t.get("grid_class", "medium")
        grid_up = bool(t.get("grid_up", True)); gen_on = bool(t.get("generator_on", False))
        power_kw = float(t["power_kw"]); solar_kw = float(t.get("solar_kw", 0)); soc = float(t.get("battery_soc_pct", 80))
        soh = float(t.get("battery_soh_pct", 90)); fuel = float(t.get("fuel_pct", 100)); cap_kwh = float(t.get("battery_capacity_kwh", 40))
        prb = float(t["prb_utilization_pct"]); sinr = float(t.get("sinr_db", 15)); drop = float(t.get("call_drop_pct", 0.3))
        setup = float(t.get("call_setup_success_pct", 99.5)); users = float(t.get("active_users", 200))
        cabinet_c = float(t.get("cabinet_temp_c", 25)); failed_logins = float(t.get("failed_logins", 0)); ip_scans = float(t.get("ip_scans", 0))
        anomaly_flag = bool(t.get("network_anomaly_flag", False))

        net_kw = max(power_kw - min(solar_kw, power_kw), 0)
        source = "SOLAR" if solar_kw >= 0.6 * power_kw else ("GENERATOR" if gen_on else ("GRID" if grid_up else "BATTERY"))
        diesel_l = net_kw * c["SFC"] * (1 + 0.3 * np.exp(-net_kw / 4)) if source == "GENERATOR" else 0.0
        grid_kwh = net_kw if source == "GRID" else 0.0
        battery_kwh = net_kw if source == "BATTERY" else 0.0
        scope1 = c["EF_DIESEL"] * diesel_l; scope2 = c["EF_GRID"] * grid_kwh
        energy_zar = grid_kwh * c["TARIFF_R_PER_KWH"] + diesel_l * c["DIESEL_R_PER_L"]
        # battery wear: draining the battery is not free even though there's no per-kWh tariff for it -
        # approximate the equivalent-cycle fraction this hour represents and cost it at BATT_WEAR_R_PER_CYCLE
        battery_wear_zar = (battery_kwh / max(cap_kwh * soh / 100, 1)) * c["BATT_WEAR_R_PER_CYCLE"]
        carbon_zar = (scope1 + scope2) * c["CARBON_SHADOW_R_PER_KG"]
        cyber_alert = (failed_logins > 15) or anomaly_flag or (ip_scans > 20)
        degradation = 1.0 if (not grid_up and not gen_on and soc <= 10) else float(np.clip((100 - setup) / 5 + max(drop - DROP_MAX, 0) / 5, 0, 1))
        revenue_at_risk = users * REVENUE_PER_USER_HOUR * degradation
        cyber_zar = c["CYBER_INCIDENT_R"] if cyber_alert else 0.0
        total_zar = energy_zar + battery_wear_zar + carbon_zar + revenue_at_risk + cyber_zar

        # battery/fuel projection (simple physics, no forecast model needed): assume load stays roughly constant
        hours = np.arange(0, 13)
        if source == "GRID" or source == "SOLAR":
            soc_proj = np.minimum(soc + hours * 3.0, 100)  # charging while grid/solar covers load
        else:
            drain_pct_per_h = net_kw / max(cap_kwh * soh / 100, 1) * 100 / c["BATT_EFF"]
            soc_proj = np.maximum(soc - hours * drain_pct_per_h, 0)
        fuel_proj = np.maximum(fuel - hours * (diesel_l / max(t.get("fuel_tank_l", 400), 1) * 100), 0) if source == "GENERATOR" else np.full_like(hours, fuel, dtype=float)

        decisions = []
        ok = sinr >= SINR_MIN and drop <= DROP_MAX
        if prb <= PRB_LOW and ok:
            frac = c["SAVE_FRAC"].get(tech, 0.06)
            decisions.append(dict(category="Energy saving", priority="medium", action=f"Enable low-load energy-saving mode ({tech.replace('_','/')} layer shutdown)",
                reason=f"PRB at {prb:.0f}% and signal quality healthy (SINR {sinr:.1f} dB, drop {drop:.2f}%)", potential_issue=None,
                est_kwh=round(frac * power_kw, 2), est_kg_co2e=round(frac * power_kw * c["EF_GRID"], 2), est_zar=round(frac * power_kw * c["TARIFF_R_PER_KWH"], 1)))
        if not grid_up and not gen_on and soc < 30 and solar_kw < 0.3 * power_kw:
            decisions.append(dict(category="Resilience", priority="high", action="Start the generator now" if fuel > 10 else "URGENT: fuel critically low - dispatch fuel immediately",
                reason=f"Grid down, battery at {soc:.0f}% and falling, solar insufficient to cover load ({solar_kw:.1f} kW vs {power_kw:.1f} kW load)", potential_issue=None,
                est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        elif not grid_up and solar_kw >= 0.5 * power_kw:
            decisions.append(dict(category="Resilience", priority="low", action="Stay on solar/battery - do not start the generator",
                reason=f"Solar ({solar_kw:.1f} kW) is covering most of the {power_kw:.1f} kW load; starting the generator now would burn diesel unnecessarily and hurt Scope 1 emissions", potential_issue=None,
                est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if fuel < 20 and grid_class in ("poor", "low"):
            decisions.append(dict(category="Fuel logistics", priority="high" if fuel < 10 else "medium", action="Schedule diesel refuel",
                reason=f"Fuel at {fuel:.0f}% on a {grid_class}-grid site", potential_issue=None, est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if cyber_alert:
            decisions.append(dict(category="Security", priority="high", action="Isolate the affected equipment and start an incident investigation",
                reason=f"{failed_logins:.0f} failed logins, {ip_scans:.0f} IP scans, anomaly flag={anomaly_flag}", potential_issue="Compromised network equipment or abnormal IoT/OT activity",
                est_kwh=0.0, est_kg_co2e=0.0, est_zar=c["CYBER_INCIDENT_R"]))
        if cabinet_c > 33:
            decisions.append(dict(category="Thermal", priority="high" if cabinet_c > 38 else "medium", action="Dispatch cooling inspection",
                reason=f"Cabinet reading {cabinet_c:.0f} C", potential_issue="Possible HVAC/fan failure or blocked airflow" if cabinet_c > 38 else None, est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))
        if soh < 75:
            decisions.append(dict(category="Maintenance", priority="medium", action="Inspect / plan battery replacement",
                reason=f"Battery SOH reading {soh:.0f}%, below the {BATTERY_SERVICE_SOH_PCT:.0f}% service threshold", potential_issue=None, est_kwh=0.0, est_kg_co2e=0.0, est_zar=-8000.0))
        if not decisions:
            decisions.append(dict(category="None", priority="low", action="No action needed", reason="All readings within normal range", potential_issue=None, est_kwh=0.0, est_kg_co2e=0.0, est_zar=0.0))

        return {
            "mode": "instant_assessment", "inputs_echo": t,
            "derived": {"power_source_inferred": source, "net_draw_kw": round(net_kw, 2), "diesel_l_this_hour": round(diesel_l, 3)},
            "financial_impact_this_hour_zar": {"energy_zar": round(energy_zar, 2), "battery_wear_zar": round(battery_wear_zar, 2), "carbon_shadow_zar": round(carbon_zar, 2),
                "revenue_at_risk_zar": round(revenue_at_risk, 2), "incident_zar": round(cyber_zar, 2), "total_zar": round(total_zar, 2),
                "degradation_factor": round(degradation, 2)},
            "esg_this_hour": {"scope1_kg_co2e": round(scope1, 3), "scope2_kg_co2e": round(scope2, 3)},
            "projection_next_12h": {"hours": hours.tolist(), "battery_soc_pct": [round(float(x), 1) for x in soc_proj], "fuel_pct": [round(float(x), 1) for x in fuel_proj]},
            "cybersecurity": {"alert": cyber_alert, "reason": "Threshold-based check (manual mode has no baseline for statistical anomaly scoring)"},
            "smart_decisions": decisions,
        }


if __name__ == "__main__":
    eng = Engine()
    for site, when in [("SITE_010", "2026-10-30 18:00"), ("SITE_402", "2026-09-15 03:00")]:
        try:
            print(json.dumps(eng.snapshot(site, when), indent=1, default=str)[:3000], "\n...\n")
        except KeyError:
            print(f"{site} not in dataset (500 sites, SITE_001..SITE_500 - check id)")
