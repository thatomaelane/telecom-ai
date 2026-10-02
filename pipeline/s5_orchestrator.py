"""Stage 5: Smart energy orchestrator (dispatch optimiser + what-if simulator).

Replays the dataset's real drivers (site load, solar, grid outages, PRB, temperatures) under two policies
and compares energy, diesel, CO2e, cost (ZAR) and network-availability risk.

BASELINE ("reactive", what the dataset itself does):
  * solar serves load only; surplus is wasted
  * battery charges from grid up to a float level; genset starts at a fixed SOC threshold (45-60 %)
SMART:
  * solar-first; surplus solar CHARGES the battery (so it is used at night instead of the generator)
  * predictive genset start: only when the battery would drop below a safe floor within the next hour
  * genset delayed while solar is covering >= half of the load
  * pre-outage battery reserve sized from site autonomy and grid-reliability class; grid charging avoids peak tariff
  * optional load levers: low-load RAN energy saving, HVAC set-point, backup-mode shedding (all QoS-guarded)
"""
import numpy as np, pandas as pd
from common import Data, save_json, MODELS, SERVING

# ---------------- constants (edit here) ----------------
CFG = dict(
    EF_GRID=0.70,             # kg CO2e/kWh, SA grid (Ember-based; Eskom does not publish a single figure). CONFIRM before final submission.
    EF_DIESEL=2.68,           # kg CO2e per litre diesel (standard factor, matches team's original dataset)
    SFC=0.32,                 # litres per kWh generated (genset), +30 % penalty at very low load (small-genset part-load derating)
    DIESEL_R_PER_L=30.05,     # SOURCED: DMPR wholesale 0.05% diesel, inland, effective 2 Sep 2026
    TARIFF_R_PER_KWH=2.80,    # ASSUMPTION: average blended Eskom Businessrate/Megaflex (peak+off-peak); +8.76% FY2027 increase applies from Apr 2026
    CARBON_SHADOW_R_PER_KG=0.308,   # SOURCED: SA headline carbon tax R308/t CO2e from 1 Jan 2026 (National Treasury). Used as an ESG SHADOW PRICE for
                                     # decision-making, not a literal per-site tax bill: the effective rate after standard allowances is only 5-48% of headline.
    BATT_WEAR_R_PER_CYCLE=1.20,     # ASSUMPTION: levelised cost per equivalent full discharge cycle, lithium/gel telecom battery
    CYBER_INCIDENT_R=5000.0,       # ASSUMPTION: average dispatch + incident-response cost per confirmed alert
    SLA_R_PER_DROPPED_PCT=500.0,   # ASSUMPTION: revenue/penalty impact per 1 pp of setup-success shortfall below 100%
    BATT_EFF=0.95,
    SAVE_FRAC={"4G": 0.04, "3G_4G": 0.07, "4G_5G": 0.08, "3G_4G_5G": 0.10},   # ASSUMPTION: share of DC load saved in low-load mode
    HVAC_SAVE_PER_C=0.04,     # ASSUMPTION: cooling energy saved per +1 C set-point
    SHED_FRAC=0.10,           # ASSUMPTION: share of DC load shed in backup mode
)
EXPECTED_OUTAGE_H = {"poor": 5.0, "low": 3.5, "medium": 2.5, "high": 2.0}
DESIGN_OUTAGE_H = {"poor": 6.0, "low": 4.0, "medium": 3.0, "high": 2.5}
BASELINE = dict(name="baseline", solar_charge=False, predictive_gen=False, gen_delay_solar=False, gen_target=88.0, gen_rate=7.0,
                float_soc=93.0, reserve_mode=False, ran_thr=None, hvac_delta=0.0, backup_shed=False)
SMART = dict(name="smart", solar_charge=True, predictive_gen=True, gen_delay_solar=True, gen_target=60.0, gen_rate=15.0,
             float_soc=None, reserve_mode=True, ran_thr=35.0, hvac_delta=0.0, backup_shed=True)


class Sim:
    def __init__(self, D, cfg=CFG):
        self.D, self.cfg = D, cfg; S, T = D.S, D.T
        a = D.a
        self.load0, self.dc, self.solar0 = a("site_total_power_kw"), a("telecom_dc_load_kw"), a("solar_generation_kw")
        self.grid_up = a("grid_voltage_v") > 0
        self.prb, self.sinr, self.drop = a("prb_utilization_pct"), a("sinr_db"), a("call_drop_rate_pct")
        self.cab, self.bank = a("cabinet_internal_temp_c"), a("battery_bank_temp_c")
        self.cool = np.maximum(self.load0 - 1.12 * self.dc - 0.15, 0)          # cooling + fan power derived from dataset
        m = D.master
        self.cap_nom = m.battery_capacity_kwh_nominal.values.astype(np.float32)
        self.soh = a("battery_soh_pct").mean(axis=1) / 100
        self.tank = m.fuel_tank_l.values.astype(np.float32)
        self.gen_thr = m.genset_start_soc_pct.values.astype(np.float32)
        self.soc0 = a("battery_soc_pct")[:, 0]; self.fuel0 = a("generator_fuel_level_pct")[:, 0]
        self.frac = np.array([cfg["SAVE_FRAC"][t] for t in m.deployed_technologies.values], np.float32)
        self.cls = m.grid_reliability_category.values
        self.hour = D.hour[:D.T]
        tou = np.where(np.isin(self.hour, [7, 8, 9, 18, 19]), 1.5, np.where((self.hour >= 22) | (self.hour < 6), 0.7, 1.0))
        self.tariff = cfg["TARIFF_R_PER_KWH"] * tou                             # ZAR/kWh by hour (TOU shape is an assumption)
        self.peak = np.isin(self.hour, [7, 8, 9, 18, 19])
        sh = lambda x: np.concatenate([x[:, :1], x[:, :-1]], axis=1)
        self.prb_p, self.sinr_p, self.drop_p, self.cab_p = sh(self.prb), sh(self.sinr), sh(self.drop), sh(self.cab)
        mean_load = self.load0.mean(axis=1)
        self.autonomy_h = self.cap_nom * self.soh * 0.85 * cfg["BATT_EFF"] / mean_load

    # ---- load levers (precomputable because they depend on observed / previous-hour data only)
    def effective_load(self, p, site_mask):
        c = self.cfg; sm = site_mask[:, None]
        ran = np.zeros_like(self.load0); hv = np.zeros_like(self.load0); shed = np.zeros_like(self.load0)
        if p["ran_thr"] is not None:
            act = (self.prb_p <= p["ran_thr"]) & (self.sinr_p >= 12) & (self.drop_p <= 1.0) & sm
            ran = self.frac[:, None] * self.dc * act
        if p["hvac_delta"] > 0:
            act = (self.cab_p <= 30) & (self.bank <= 30) & (self.cool > 0.3) & sm
            hv = self.cool * min(c["HVAC_SAVE_PER_C"] * p["hvac_delta"], 0.4) * act
        if p["backup_shed"]:
            shed = c["SHED_FRAC"] * self.dc * ((~self.grid_up) & (self.prb_p <= 60) & sm)
        return np.maximum(self.load0 - ran - hv - shed, 0.3)

    def run(self, p, site_mask=None, batt_scale=1.0, solar_scale=1.0):
        c = self.cfg; S, T = self.load0.shape
        site_mask = np.ones(S, bool) if site_mask is None else site_mask
        # sites outside the mask keep the BASELINE policy; sites inside get p
        load = np.where(site_mask[:, None], self.effective_load(p, site_mask), self.load0)
        solar = self.solar0 * solar_scale
        cap = self.cap_nom * self.soh * batt_scale
        eta, sfc = c["BATT_EFF"], c["SFC"]
        mean_net = np.maximum(load.mean(axis=1) - solar.mean(axis=1) * 0.3, 0.5)
        reserve = np.clip(25 + np.array([EXPECTED_OUTAGE_H[k] for k in self.cls]) * mean_net / (cap * eta) * 100, 30, 100)
        soc, fuel, gen_on = self.soc0.astype(np.float32).copy(), self.fuel0.astype(np.float32).copy(), np.zeros(S, bool)
        O = {k: np.zeros((S, T), np.float32) for k in ["grid", "diesel", "gen", "soc", "sol_used", "sol_curt", "risk", "cost_grid"]}
        for t in range(T):
            up, sol, ld = self.grid_up[:, t], solar[:, t], load[:, t]
            smart = site_mask & p["reserve_mode"]                                # per-site policy switch
            used = np.minimum(sol, ld); surplus = sol - used; net = ld - used
            sc = smart & p["solar_charge"]
            chg_sol = np.where(sc, np.minimum(np.minimum(surplus * eta / cap * 100, 100 - soc), 25.0), 0.0)
            sol_to_batt = chg_sol * cap / 100 / eta
            # ---------- grid up
            target = np.where(smart, reserve, BASELINE["float_soc"])
            rate = np.where(soc < 80, 9.0, 3.0)
            rate = np.where(smart, np.where(soc < reserve, 12.0, 0.0), rate)
            no_peak = ~(smart & self.peak[t] & (soc >= 30))
            chg_grid = np.where(no_peak, np.clip(np.minimum(rate - chg_sol, target - soc - chg_sol), 0, None), 0.0)
            soc_up = np.minimum(soc + chg_sol + chg_grid, 100.0)
            grid_kwh_up = net + chg_grid * cap / 100 / eta
            # ---------- grid down
            drain = net / cap * 100 / eta
            start_thr = np.where(smart & p["predictive_gen"], 20.0 + drain, self.gen_thr)
            can_delay = smart & p["gen_delay_solar"] & (sol >= 0.5 * ld) & (soc > 35)   # never delay below a safety floor
            start = (~up) & (soc <= start_thr) & (fuel > 2.5) & ~can_delay
            tgt_gen = np.where(smart, p["gen_target"], BASELINE["gen_target"])
            gen_on = ((gen_on | start) & (soc < tgt_gen) & (fuel > 2.5)) & ~up
            g_rate = np.where(smart, p["gen_rate"], BASELINE["gen_rate"])
            chg_gen = np.where(gen_on, np.minimum(g_rate, tgt_gen - soc), 0.0)
            gen_kwh = np.where(gen_on, net + chg_gen * cap / 100 / eta, 0.0)
            diesel = gen_kwh * sfc * (1 + 0.3 * np.exp(-gen_kwh / 4))
            soc_dn = np.where(gen_on, soc + chg_gen + chg_sol, np.maximum(soc - drain, 5.0) + chg_sol)
            soc_new = np.minimum(np.where(up, soc_up, soc_dn), 100.0)
            fuel = np.maximum(fuel - diesel / self.tank * 100, 0)
            if self.hour[t] == 9:
                fuel = np.where((fuel < 30) & (self.D.static.cat.values <= 1), 95.0, fuel)   # refuel logistics on poor/low grid
            grid_kwh = np.where(up, grid_kwh_up, 0.0)
            O["grid"][:, t] = grid_kwh; O["diesel"][:, t] = diesel; O["gen"][:, t] = gen_on; O["soc"][:, t] = soc_new
            O["sol_used"][:, t] = used + np.where(sc, sol_to_batt, 0.0); O["sol_curt"][:, t] = sol - O["sol_used"][:, t]
            O["risk"][:, t] = (~up) & (~gen_on) & (soc_new <= 15); O["cost_grid"][:, t] = grid_kwh * self.tariff[t]
            soc = soc_new
        O["load"] = load
        return O

    def summarise(self, O, days):
        c = self.cfg; S, T = O["load"].shape
        g, dsl = O["grid"].sum(), O["diesel"].sum()
        s2, s1 = c["EF_GRID"] * g, c["EF_DIESEL"] * dsl
        cost = O["cost_grid"].sum() + c["DIESEL_R_PER_L"] * dsl
        k = 30.0 / days
        return dict(energy_consumed_mwh=float(O["load"].sum() / 1e3), grid_mwh=float(g / 1e3), diesel_l=float(dsl), gen_hours=int(O["gen"].sum()),
                    scope1_t=float(s1 / 1e3), scope2_t=float(s2 / 1e3), co2e_t=float((s1 + s2) / 1e3), cost_zar_m=float(cost / 1e6),
                    solar_used_mwh=float(O["sol_used"].sum() / 1e3), solar_wasted_mwh=float(O["sol_curt"].sum() / 1e3),
                    risk_hours=int(O["risk"].sum()), availability_pct=float(100 * (1 - O["risk"].sum() / (S * T))),
                    renewable_share_pct=float(100 * O["sol_used"].sum() / max(O["load"].sum(), 1)),
                    monthly_cost_zar_m=float(cost / 1e6 * k), monthly_co2e_t=float((s1 + s2) / 1e3 * k))


def site_classes(sim, base, smart):
    """Availability-risk class per site, and Optimise / Review / Don't-optimise call for load-reducing levers."""
    d_h = np.array([DESIGN_OUTAGE_H[k] for k in sim.cls])
    cover = sim.autonomy_h / d_h
    base_risk = base["risk"].sum(axis=1)
    risk = np.where((cover < 0.8) | ((base_risk > 0) & np.isin(sim.cls, ["poor", "low"])), "High", np.where(cover < 1.3, "Medium", "Low"))
    sav = (sim.load0.sum(axis=1) - smart["load"].sum(axis=1)) / sim.load0.sum(axis=1) * 100
    call = np.select([risk == "Low", risk == "Medium"], ["Optimise", "Review"], "Do not optimise")
    return risk, sav, call, cover


def main(days=60):
    D = Data(); sim = Sim(D); S, T = D.S, D.T
    print("dataset check -> actual diesel in data:", round(D.df.diesel_consumed_liters.sum()), "L | actual grid-down hours:", int((~sim.grid_up).sum()))
    B = sim.run(BASELINE); Sm = sim.run(SMART)
    sb, ss = sim.summarise(B, days), sim.summarise(Sm, days)
    rep = {"baseline": sb, "smart": ss, "config": CFG, "window_days": days}
    rep["delta"] = {k: ss[k] - sb[k] for k in sb}
    rep["sensitivity_EF_grid"] = {}
    for ef in (0.70, 0.95):
        c2 = dict(CFG, EF_GRID=ef); sm2 = Sim(D, c2); a, b = sm2.summarise(B, days), sm2.summarise(Sm, days)
        rep["sensitivity_EF_grid"][str(ef)] = {"baseline_co2e_t": a["co2e_t"], "smart_co2e_t": b["co2e_t"], "saved_t": a["co2e_t"] - b["co2e_t"]}
    # ---- attribution: turn levers on one by one
    steps = [("A. solar surplus -> battery", dict(BASELINE, solar_charge=True, reserve_mode=True, predictive_gen=False, gen_delay_solar=False, gen_target=88.0, gen_rate=7.0, ran_thr=None, backup_shed=False)),
             ("B. + predictive genset start", dict(BASELINE, solar_charge=True, reserve_mode=True, predictive_gen=True, gen_delay_solar=False, gen_target=60.0, gen_rate=15.0, ran_thr=None, backup_shed=False)),
             ("C. + delay genset while solar covers >=50%", dict(SMART, ran_thr=None, backup_shed=False)),
             ("D. + backup-mode shedding", dict(SMART, ran_thr=None)), ("E. + low-load RAN saving (full smart)", SMART)]
    rep["attribution"] = []
    for nm, pp in steps:
        s = sim.summarise(sim.run(pp), days)
        rep["attribution"].append({"step": nm, "diesel_l": s["diesel_l"], "co2e_t": s["co2e_t"], "cost_zar_m": s["cost_zar_m"], "availability_pct": s["availability_pct"], "risk_hours": s["risk_hours"]})
    risk, sav, call, cover = site_classes(sim, B, Sm)
    sites = pd.DataFrame({"site_id": D.sites, "grid_class": sim.cls, "battery_autonomy_h": sim.autonomy_h.round(2), "autonomy_vs_design_outage": cover.round(2),
        "availability_risk": risk, "energy_saving_pct": sav.round(2), "decision": call,
        "baseline_diesel_l": B["diesel"].sum(1), "smart_diesel_l": Sm["diesel"].sum(1),
        "baseline_co2e_kg": CFG["EF_GRID"] * B["grid"].sum(1) + CFG["EF_DIESEL"] * B["diesel"].sum(1),
        "smart_co2e_kg": CFG["EF_GRID"] * Sm["grid"].sum(1) + CFG["EF_DIESEL"] * Sm["diesel"].sum(1),
        "baseline_risk_hours": B["risk"].sum(1), "smart_risk_hours": Sm["risk"].sum(1),
        "baseline_solar_wasted_kwh": B["sol_curt"].sum(1), "smart_solar_wasted_kwh": Sm["sol_curt"].sum(1)})
    sites["co2e_saved_kg"] = sites.baseline_co2e_kg - sites.smart_co2e_kg
    sites.to_parquet(SERVING / "orchestrator_sites.parquet", index=False)
    rep["decision_counts"] = sites.decision.value_counts().to_dict(); rep["risk_counts"] = sites.availability_risk.value_counts().to_dict()
    day = np.repeat(np.arange(days), 24)
    daily = pd.DataFrame({"day": pd.date_range(D.ts[0], periods=days, freq="D").date})
    for nm, O in (("baseline", B), ("smart", Sm)):
        daily[nm + "_co2e_t"] = [(CFG["EF_GRID"] * O["grid"][:, day == d].sum() + CFG["EF_DIESEL"] * O["diesel"][:, day == d].sum()) / 1e3 for d in range(days)]
        daily[nm + "_diesel_l"] = [O["diesel"][:, day == d].sum() for d in range(days)]
        daily[nm + "_cost_zar"] = [O["cost_grid"][:, day == d].sum() + CFG["DIESEL_R_PER_L"] * O["diesel"][:, day == d].sum() for d in range(days)]
    daily.to_parquet(SERVING / "orchestrator_daily.parquet", index=False)
    # ---- showcase timeline: a poor-grid site with the most baseline diesel avoided
    cand = sites[sites.grid_class.isin(["poor", "low"])].assign(gain=lambda x: x.baseline_diesel_l - x.smart_diesel_l).sort_values("gain", ascending=False).head(6).site_id
    rows = []
    for sid in cand:
        i = int(np.where(D.sites == sid)[0][0])
        rows.append(pd.DataFrame({"site_id": sid, "time": D.ts[:T], "load_kw": sim.load0[i], "solar_kw": sim.solar0[i], "grid_up": sim.grid_up[i],
            "baseline_soc": B["soc"][i], "smart_soc": Sm["soc"][i], "baseline_gen": B["gen"][i], "smart_gen": Sm["gen"][i]}))
    pd.concat(rows).to_parquet(SERVING / "orchestrator_timeline_examples.parquet", index=False)
    hourly = pd.DataFrame({
        "site_id": np.repeat(D.sites, T), "timestamp": np.tile(D.ts[:T].values, S),
        "load_kw": sim.load0.ravel(), "solar_kw": sim.solar0.ravel(), "grid_up": sim.grid_up.ravel(),
        "baseline_grid_kwh": B["grid"].ravel(), "baseline_diesel_l": B["diesel"].ravel(), "baseline_gen_on": B["gen"].ravel().astype(bool),
        "baseline_soc": B["soc"].ravel(), "baseline_cost_grid_zar": B["cost_grid"].ravel(), "baseline_risk": B["risk"].ravel().astype(bool),
        "smart_grid_kwh": Sm["grid"].ravel(), "smart_diesel_l": Sm["diesel"].ravel(), "smart_gen_on": Sm["gen"].ravel().astype(bool),
        "smart_soc": Sm["soc"].ravel(), "smart_cost_grid_zar": Sm["cost_grid"].ravel(), "smart_risk": Sm["risk"].ravel().astype(bool),
    })
    hourly["baseline_co2e_kg"] = CFG["EF_GRID"] * hourly.baseline_grid_kwh + CFG["EF_DIESEL"] * hourly.baseline_diesel_l
    hourly["smart_co2e_kg"] = CFG["EF_GRID"] * hourly.smart_grid_kwh + CFG["EF_DIESEL"] * hourly.smart_diesel_l
    hourly["baseline_cost_zar"] = hourly.baseline_cost_grid_zar + CFG["DIESEL_R_PER_L"] * hourly.baseline_diesel_l
    hourly["smart_cost_zar"] = hourly.smart_cost_grid_zar + CFG["DIESEL_R_PER_L"] * hourly.smart_diesel_l
    for c in hourly.columns:
        if hourly[c].dtype == np.float64: hourly[c] = hourly[c].astype(np.float32)
    hourly.to_parquet(SERVING / "orchestrator_hourly.parquet", index=False, compression="zstd")
    print("saved orchestrator_hourly.parquet", hourly.shape)
    save_json(rep, MODELS / "report_orchestrator.json")
    for k in ("baseline", "smart", "delta"):
        print(k, {a: round(b, 2) for a, b in rep[k].items()})
    print("attribution:"); [print("  ", r) for r in rep["attribution"]]
    print("sensitivity:", rep["sensitivity_EF_grid"]); print("decisions:", rep["decision_counts"], "| risk classes:", rep["risk_counts"])


if __name__ == "__main__":
    main()
