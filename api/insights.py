"""Customer-experience, cost, energy-mix (with battery), live-feed and fleet-forecast helpers.

Added in the "final improvements" release. Everything here is built ONCE at API start-up from the objects
the API already loads (the query Engine, the serving parquet tables) - no model is retrained and no
pipeline file is changed.

Terminology used across the dashboard (kept identical in the UI):
  CURRENT VALUES   - recorded telemetry for the latest hour (plus the 2-minute simulated live feed on top)
  PREDICTED VALUES - output of the forecasting / outage-risk models for the next hours
  MODEL DECISION   - what the AI recommends doing, with the numbers behind it
"""
import time
import numpy as np
import pandas as pd

from s5_orchestrator import CFG
from s6_query import REVENUE_PER_USER_HOUR, DROP_MAX, ARPU_R_PER_MONTH

LIVE_PERIOD_S = 120            # the "live" readings move every 2 minutes
THEFT_DAMAGE_R = 15000.0       # ASSUMPTION: replacement of stolen fuel/battery + call-out, per theft incident
THERMAL_DAMAGE_R_PER_H = 150.0 # ASSUMPTION: equipment-stress cost per hour the cabinet is above THERMAL_C
THERMAL_C = 38.0
INCIDENT_GAP_H = 3             # alert-hours closer than this (same site) count as ONE incident
COMPONENTS = ["tariff", "diesel", "damages", "esg", "penalties", "lost_revenue", "cyber"]
SEV_RANK = {"critical": 0, "high": 1, "medium": 2}


# ------------------------------------------------------------------ small helpers
def sanitize(o):
    """Make any nested structure JSON-safe (numpy scalars -> python, NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


def cx_score(drop, setup, dl, lat, sinr):
    """0-100 customer-experience score from the dataset's own CX metrics (call drop, call set-up success,
    downlink throughput, latency, SINR). Weights are a documented assumption, shown on the About page."""
    drop, setup, dl, lat, sinr = (np.asarray(x, float) for x in (drop, setup, dl, lat, sinr))
    s = (0.30 * np.clip(1 - drop / 2.0, 0, 1) + 0.25 * np.clip((setup - 92.0) / 7.5, 0, 1)
         + 0.20 * np.clip(dl / 60.0, 0, 1) + 0.15 * np.clip(1 - (lat - 15.0) / 45.0, 0, 1)
         + 0.10 * np.clip((sinr - 5.0) / 10.0, 0, 1))
    return np.round(100 * s, 1)


def cx_label(score):
    return np.select([np.asarray(score) >= 80, np.asarray(score) >= 60], ["Good", "Fair"], "Poor")


def net_label(prb, sinr):
    prb, sinr = np.asarray(prb, float), np.asarray(sinr, float)
    return np.select([(prb >= 85) | (sinr < 8), (prb >= 65) | (sinr < 12)], ["Congested", "Busy"], "Healthy")


def alignment_text(net, cx):
    """One plain-language line that ties the customer experience to the network performance."""
    if net == "Healthy" and cx == "Good":
        return "Aligned: healthy network, good customer experience."
    if net == "Healthy":
        return "Check: network load is healthy but experience is not good - look at signal quality or equipment."
    if net == "Busy" and cx != "Poor":
        return "Aligned: network is busy and experience is holding up."
    if net == "Busy":
        return "Aligned: rising load or weak signal is pulling experience down."
    if cx == "Good":
        return "Network is congested but experience is still good - watch closely."
    return "Aligned: congestion / weak signal is degrading customer experience."


def _ns(idx):
    return pd.DatetimeIndex(pd.to_datetime(idx)).as_unit("ns")


def _incident_starts(m):
    """(S,T) bool alert-hours -> bool starts of incidents (an alert with none in the previous INCIDENT_GAP_H hours)."""
    prev = np.zeros_like(m)
    for k in range(1, INCIDENT_GAP_H + 1):
        shifted = np.zeros_like(m)
        shifted[:, k:] = m[:, :-k]
        prev |= shifted
    return m & ~prev


class Insights:
    def __init__(self, engine, site_status, cyber, forecast=None):
        self.engine = engine
        D = engine.D
        self.D = D
        S, T = D.S, D.T
        self.S, self.T = S, T
        a = D.a
        # ---- recorded telemetry matrices (sites x hours)
        self.prb = a("prb_utilization_pct"); self.users = a("active_rrc_connected_users"); self.vol = a("data_volume_gb")
        self.dl = a("avg_dl_throughput_mbps"); self.ul = a("avg_ul_throughput_mbps"); self.lat = a("latency_rtt_ms")
        self.setup = a("call_setup_success_rate_pct"); self.drop = a("call_drop_rate_pct")
        self.sinr = a("sinr_db"); self.rsrp = a("rsrp_dbm"); self.cab = a("cabinet_internal_temp_c")
        self.tamper = a("physical_tamper_sensor_status"); self.volt = a("grid_voltage_v")
        self.cx = cx_score(self.drop, self.setup, self.dl, self.lat, self.sinr).astype(np.float32)

        # ---- orchestrator (AI-managed) hourly matrices, in the same site order as the dataset
        h = engine.hourly
        order = np.lexsort((h.timestamp.to_numpy(), h.site_id.map(engine.site_idx).to_numpy()))   # site, then time (no table copy)

        def m(col, dtype=np.float32):
            return h[col].to_numpy()[order].reshape(S, T).astype(dtype)
        self.load = m("load_kw"); self.solar = m("solar_kw"); self.grid_up = m("grid_up", bool)
        self.s_gen = m("smart_gen_on", bool); self.s_soc = m("smart_soc"); self.s_risk = m("smart_risk", bool)
        self.s_diesel = m("smart_diesel_l"); self.s_cost_grid = m("smart_cost_grid_zar")
        self.s_cost = m("smart_cost_zar"); self.s_co2 = m("smart_co2e_kg")
        self.b_cost = m("baseline_cost_zar"); self.b_co2 = m("baseline_co2e_kg")

        self.cyber = cyber
        self._build_alert_matrices(cyber)
        self._build_costs()
        self._fit_cx_curves()
        self.site_status = site_status
        self._build_base(site_status, cyber)
        self.forecast_cache = self._build_fleet_forecast(forecast, site_status)
        self.phase = np.random.default_rng(7).uniform(0, 2 * np.pi, S)

    # ------------------------------------------------------------------ alerts -> (S,T) matrices
    def _build_alert_matrices(self, cyber):
        S, T, D = self.S, self.T, self.D
        al = np.zeros((S, T), bool); th = np.zeros((S, T), bool)
        if cyber is not None and len(cyber):
            si = cyber.site_id.map(self.engine.site_idx)
            ti = _ns(D.ts[:T]).get_indexer(_ns(cyber.time))
            ok = si.notna().to_numpy() & (ti >= 0)
            s_ok = si.to_numpy()[ok].astype(int); t_ok = ti[ok]
            al[s_ok, t_ok] = True
            is_theft = cyber.reason.astype(str).str.contains("theft", case=False).to_numpy()[ok]
            th[s_ok[is_theft], t_ok[is_theft]] = True
        self.alert = al; self.theft = th

    # ------------------------------------------------------------------ cost model
    def _build_costs(self):
        c = CFG
        prev = np.concatenate([self.s_soc[:, :1], self.s_soc[:, :-1]], axis=1)
        battery = np.abs(self.s_soc - prev) / 100.0 * c["BATT_WEAR_R_PER_CYCLE"]
        thermal = (self.cab > THERMAL_C) * THERMAL_DAMAGE_R_PER_H
        theft = _incident_starts(self.theft) * THEFT_DAMAGE_R
        viol = (self.drop > DROP_MAX) | (self.setup < 95.0)
        degr = np.where(self.s_risk, 1.0, np.clip((100 - self.setup) / 5 + np.maximum(self.drop - DROP_MAX, 0) / 5, 0, 1))
        self.comp = {
            "tariff": self.s_cost_grid.astype(np.float32),
            "diesel": (self.s_diesel * c["DIESEL_R_PER_L"]).astype(np.float32),
            "damages": (battery + thermal + theft).astype(np.float32),
            "esg": (self.s_co2 * c["CARBON_SHADOW_R_PER_KG"]).astype(np.float32),
            "penalties": np.where(viol, (100 - self.setup) * c["SLA_R_PER_DROPPED_PCT"], 0.0).astype(np.float32),
            "lost_revenue": (self.users * REVENUE_PER_USER_HOUR * degr).astype(np.float32),
            "cyber": (_incident_starts(self.alert) * c["CYBER_INCIDENT_R"]).astype(np.float32),
        }
        # split of the "damages" line, for the explanation tooltip
        self.dmg_parts = {"battery_wear": battery.astype(np.float32), "thermal": thermal.astype(np.float32), "theft": theft.astype(np.float32)}
        self.total = sum(self.comp.values())

    def assumptions(self):
        c = CFG
        return [
            {"name": "Grid tariff", "value": f"R{c['TARIFF_R_PER_KWH']:.2f}/kWh base, time-of-use (x1.5 peak 07-09h & 18-19h, x0.7 night)", "kind": "assumption"},
            {"name": "Diesel", "value": f"R{c['DIESEL_R_PER_L']:.2f}/litre (DMPR wholesale 0.05% diesel, inland, 2 Sep 2026)", "kind": "sourced"},
            {"name": "ESG / carbon", "value": f"R{c['CARBON_SHADOW_R_PER_KG'] * 1000:.0f}/t CO2e shadow price (SA headline carbon tax from 1 Jan 2026)", "kind": "sourced"},
            {"name": "SLA / QoS penalty", "value": f"R{c['SLA_R_PER_DROPPED_PCT']:.0f} per 1 pp set-up-success shortfall, in any hour with drop > {DROP_MAX:.1f}% or set-up < 95%", "kind": "assumption"},
            {"name": "Lost revenue (unmet demand)", "value": f"Active users x R{REVENUE_PER_USER_HOUR:.3f}/user/hour (ARPU R{ARPU_R_PER_MONTH:.0f}/month) x service-degradation factor (100% when a site is out of power)", "kind": "assumption"},
            {"name": "Damages", "value": f"Theft R{THEFT_DAMAGE_R:,.0f} per incident + thermal stress R{THERMAL_DAMAGE_R_PER_H:.0f}/h above {THERMAL_C:.0f} C + battery wear R{c['BATT_WEAR_R_PER_CYCLE']:.2f}/cycle", "kind": "assumption"},
            {"name": "Cyber response", "value": f"R{c['CYBER_INCIDENT_R']:,.0f} per incident (alert-hours within {INCIDENT_GAP_H} h at one site count as one incident)", "kind": "assumption"},
        ]

    def cost_summary(self, hours=24):
        T, S, D = self.T, self.S, self.D
        hours = int(np.clip(hours, 24, 168))
        step = 1 if hours <= 48 else 24
        hours = (hours // step) * step
        sl = slice(T - hours, T)
        comp = {k: float(v[:, sl].sum()) for k, v in self.comp.items()}
        total = float(sum(comp.values()))
        n = hours // step
        series = {k: np.round(v[:, sl].sum(axis=0).reshape(n, step).sum(axis=1), 0).tolist() for k, v in self.comp.items()}
        ts = D.ts[T - hours:T]
        labels = [ts[j * step].strftime("%H:00") if step == 1 else ts[j * step].strftime("%m-%d") for j in range(n)]
        shadow = CFG["CARBON_SHADOW_R_PER_KG"]
        base_energy = float(self.b_cost[:, sl].sum() + self.b_co2[:, sl].sum() * shadow)
        smart_energy = float(self.s_cost[:, sl].sum() + self.s_co2[:, sl].sum() * shadow)
        cls = self.D.master.grid_reliability_category.to_numpy()
        by_class = []
        for g in ["poor", "low", "medium", "high"]:
            mk = cls == g
            if not mk.any():
                continue
            row = {k: float(v[mk][:, sl].sum()) for k, v in self.comp.items()}
            by_class.append({"grid_class": g, "sites": int(mk.sum()), **row, "total": float(sum(row.values()))})
        dmg = {k: float(v[:, sl].sum()) for k, v in self.dmg_parts.items()}
        return sanitize({
            "hours": hours, "from": str(D.ts[T - hours]), "to": str(D.ts[T - 1]), "components": comp, "total_zar": total,
            "operating_zar": comp["tariff"] + comp["diesel"], "risk_penalty_zar": total - comp["tariff"] - comp["diesel"],
            "damages_split": dmg, "series": {"labels": labels, **series}, "by_grid_class": by_class,
            "ai_vs_baseline": {"baseline_energy_carbon_zar": base_energy, "ai_energy_carbon_zar": smart_energy, "saving_zar": base_energy - smart_energy,
                               "saving_pct": (base_energy - smart_energy) / base_energy * 100 if base_energy > 0 else 0.0},
            "incidents": {"cyber": int(_incident_starts(self.alert)[:, sl].sum()), "theft": int(_incident_starts(self.theft)[:, sl].sum()),
                          "thermal_hours": int((self.cab[:, sl] > THERMAL_C).sum()), "sla_violation_hours": int((self.comp["penalties"][:, sl] > 0).sum()),
                          "site_hours_without_power": int(self.s_risk[:, sl].sum())},
            "assumptions": self.assumptions(),
        })

    def cost_sites(self, hours=24, sort="total", q=None, limit=50):
        T, D = self.T, self.D
        hours = int(np.clip(hours, 24, 168)); sl = slice(T - hours, T)
        df = pd.DataFrame({"site_id": D.sites, "grid_class": D.master.grid_reliability_category.to_numpy(), "tier": D.master.site_demographic_tier.to_numpy()})
        for k, v in self.comp.items():
            df[k] = v[:, sl].sum(axis=1)
        df["total"] = df[COMPONENTS].sum(axis=1)
        if q:
            df = df[df.site_id.str.contains(str(q).strip(), case=False, regex=False)]
        if sort not in df.columns:
            sort = "total"
        df = df.sort_values(sort, ascending=False)
        n_total = len(df)
        df = df.head(int(np.clip(limit, 1, 500)))
        num = [c for c in df.columns if c not in ("site_id", "grid_class", "tier")]
        df[num] = df[num].round(0)
        return sanitize({"hours": hours, "matched": n_total, "rows": df.to_dict(orient="records")})

    # ------------------------------------------------------------------ energy source mix, battery included
    def _mix_from(self, load, solar, up, gen):
        solar_direct = np.minimum(solar, load)
        net = load - solar_direct
        grid = np.where(up, net, 0.0)
        diesel = np.where(~up & gen, net, 0.0)
        battery = np.where(~up & ~gen, net, 0.0)
        parts = {"grid_kwh": float(grid.sum()), "diesel_generator_kwh": float(diesel.sum()), "solar_kwh": float(solar_direct.sum()), "battery_kwh": float(battery.sum())}
        tot = sum(parts.values()) or 1.0
        out = {k: round(v, 1) for k, v in parts.items()}
        out.update({"grid_pct": round(parts["grid_kwh"] / tot * 100, 1), "diesel_pct": round(parts["diesel_generator_kwh"] / tot * 100, 1),
                    "solar_pct": round(parts["solar_kwh"] / tot * 100, 1), "battery_pct": round(parts["battery_kwh"] / tot * 100, 1), "total_kwh": round(tot, 1)})
        return out

    def mix(self, hours=24):
        """Energy that actually carried the load, by source: grid, diesel generator, solar PV and BATTERY
        (battery = stored energy discharged during outages; it was charged earlier from grid/solar/diesel)."""
        hours = int(np.clip(hours, 1, self.T)); sl = slice(self.T - hours, self.T)
        out = self._mix_from(self.load[:, sl], self.solar[:, sl], self.grid_up[:, sl], self.s_gen[:, sl])
        out["hours"] = hours
        return out

    # ------------------------------------------------------------------ fleet CX series (last N hours)
    def cx_fleet(self, hours=72):
        T, D = self.T, self.D
        hours = int(np.clip(hours, 6, 24 * 14)); sl = slice(T - hours, T)

        def mean(x):
            return np.round(x[:, sl].mean(axis=0), 2).tolist()
        series = {"timestamps": [str(t) for t in D.ts[T - hours:T]], "call_drop_pct": mean(self.drop), "call_setup_success_pct": mean(self.setup),
                  "dl_mbps": mean(self.dl), "ul_mbps": mean(self.ul), "latency_ms": mean(self.lat), "sinr_db": mean(self.sinr),
                  "prb_pct": mean(self.prb), "cx_score": mean(self.cx),
                  "sites_poor_pct": np.round((self.cx[:, sl] < 60).mean(axis=0) * 100, 1).tolist()}
        return sanitize({"hours": hours, "series": series})

    # ------------------------------------------------------------------ CX vs load curves (used for the predicted CX)
    def _fit_cx_curves(self):
        T = self.T; sl = slice(max(0, T - 24 * 14), T)
        rng = np.random.default_rng(0)
        x = self.prb[:, sl].ravel()
        idx = rng.choice(len(x), size=min(200000, len(x)), replace=False)
        self._arr = {"dl": self.dl, "drop": self.drop, "lat": self.lat, "setup": self.setup, "sinr": self.sinr}
        self.curves = {k: np.polyfit(x[idx], v[:, sl].ravel()[idx], 2) for k, v in self._arr.items()}
        # per-site offset = how far each site sits from the fleet curve (mean over the last 24 h)
        w = slice(max(0, T - 24), T)
        self.offset = {k: (v[:, w] - np.polyval(self.curves[k], self.prb[:, w])).mean(axis=1) for k, v in self._arr.items()}

    def _predict_cx_from_prb(self, prb_pred, offsets):
        prb_pred = np.asarray(prb_pred, float)
        dl = np.clip(np.polyval(self.curves["dl"], prb_pred) + offsets["dl"], 1, None)
        drop = np.clip(np.polyval(self.curves["drop"], prb_pred) + offsets["drop"], 0.05, None)
        lat = np.clip(np.polyval(self.curves["lat"], prb_pred) + offsets["lat"], 5, None)
        setup = np.clip(np.polyval(self.curves["setup"], prb_pred) + offsets["setup"], 80, 100)
        sinr = np.clip(np.polyval(self.curves["sinr"], prb_pred) + offsets["sinr"], 0, 35)
        return dl, drop, lat, setup, sinr

    # ------------------------------------------------------------------ per-site CX (Site Explorer)
    def site_cx(self, i, p, prb_forecast=None, hist_hours=24):
        a0 = max(0, p - hist_hours + 1)
        sl = slice(a0, p + 1)
        hist = {"timestamps": [str(t) for t in self.D.ts[a0:p + 1]], "dl_mbps": np.round(self.dl[i, sl], 1).tolist(), "latency_ms": np.round(self.lat[i, sl], 1).tolist(),
                "call_drop_pct": np.round(self.drop[i, sl], 2).tolist(), "call_setup_success_pct": np.round(self.setup[i, sl], 2).tolist(),
                "sinr_db": np.round(self.sinr[i, sl], 1).tolist(), "prb_pct": np.round(self.prb[i, sl], 1).tolist(), "cx_score": np.round(self.cx[i, sl], 1).tolist()}
        score_now = float(self.cx[i, p]); prb_now = float(self.prb[i, p]); sinr_now = float(self.sinr[i, p])
        cl = str(cx_label(score_now)); nl = str(net_label(prb_now, sinr_now))
        out = {"score": round(score_now, 1), "label": cl, "network_status": nl, "alignment": alignment_text(nl, cl), "history_24h": hist, "forecast": None}
        if prb_forecast is not None and len(prb_forecast):
            off = {k: float(v[i]) for k, v in self.offset.items()}
            dl, drop, lat, setup, sinr = self._predict_cx_from_prb(prb_forecast, off)
            sc = cx_score(drop, setup, dl, lat, sinr)
            out["forecast"] = {"dl_mbps": np.round(dl, 1).tolist(), "call_drop_pct": np.round(drop, 2).tolist(), "latency_ms": np.round(lat, 1).tolist(),
                               "call_setup_success_pct": np.round(setup, 2).tolist(), "cx_score": np.round(sc, 1).tolist(),
                               "worst_label": str(cx_label(sc.min())), "worst_score": round(float(sc.min()), 1)}
        c0 = max(0, p - 23)
        out["cost_24h"] = {k: round(float(v[i, c0:p + 1].sum()), 0) for k, v in self.comp.items()}
        out["cost_24h"]["total"] = round(sum(out["cost_24h"].values()), 0)
        return sanitize(out)

    def site_history_extra(self, i, p0, p1):
        sl = slice(p0, p1 + 1)
        return sanitize({"dl_mbps": np.round(self.dl[i, sl], 1), "latency_ms": np.round(self.lat[i, sl], 1), "sinr_db": np.round(self.sinr[i, sl], 1),
                         "prb_pct": np.round(self.prb[i, sl], 1), "cx_score": np.round(self.cx[i, sl], 1)})

    # ------------------------------------------------------------------ latest-hour table for every site (base of the live feed)
    def _build_base(self, site_status, cyber):
        D, T, p = self.D, self.T, self.T - 1
        st = site_status.set_index("site_id").loc[D.sites]
        b = pd.DataFrame({"site_id": D.sites})
        for col in ["lat", "lon", "tier", "grid_class", "source", "soc", "soh", "cab_temp", "prb", "power_kw", "solar_kw", "fuel", "outage_risk_3h", "status"]:
            b[col] = st[col].to_numpy()
        b["users"] = self.users[:, p]; b["vol"] = self.vol[:, p]; b["dl"] = self.dl[:, p]; b["ul"] = self.ul[:, p]
        b["latency_ms"] = self.lat[:, p]; b["setup"] = self.setup[:, p]; b["drop"] = self.drop[:, p]
        b["sinr"] = self.sinr[:, p]; b["rsrp"] = self.rsrp[:, p]
        b["grid_up"] = self.volt[:, p] > 0
        b["tamper"] = self.tamper[:, p] > 0
        b["cost24h"] = sum(v[:, T - 24:T].sum(axis=1) for v in self.comp.values())
        # cyber & physical alerts, last 24 h of the dataset
        b["cyber_n"] = 0; b["theft_n"] = 0; b["cyber_text"] = ""
        if cyber is not None and len(cyber):
            c24 = cyber[pd.to_datetime(cyber.time) >= D.ts[-1] - pd.Timedelta(hours=24)].copy()
            if len(c24):
                c24["_r"] = c24.severity.map(SEV_RANK).fillna(3)
                is_theft = c24.reason.astype(str).str.contains("theft", case=False)
                idx = {s: k for k, s in enumerate(D.sites)}
                for sid, g in c24.groupby("site_id"):
                    k = idx.get(sid)
                    if k is None:
                        continue
                    top = g.sort_values(["_r", "time"], ascending=[True, False]).iloc[0]
                    b.loc[k, "cyber_n"] = int((~is_theft.loc[g.index]).sum())
                    b.loc[k, "theft_n"] = int(is_theft.loc[g.index].sum())
                    b.loc[k, "cyber_text"] = f"{top.reason} ({top.severity}, {len(g)} alert-hours in 24 h)"
        self.base = b

    # ------------------------------------------------------------------ live feed
    def live(self, now=None):
        """Latest recorded hour + a smooth simulated variation that changes every LIVE_PERIOD_S seconds.
        Load-driven values move together (PRB, users, power) and the customer-experience metrics follow the
        load, so network performance and customer experience always stay aligned."""
        now = time.time() if now is None else now
        bucket = int(now // LIVE_PERIOD_S)
        b = self.base
        S = self.S
        rng = np.random.default_rng(bucket)
        f = np.clip(1 + 0.07 * np.sin(2 * np.pi * bucket / 12.0 + self.phase) + 0.025 * rng.standard_normal(S), 0.75, 1.30)
        prb = np.clip(b.prb.to_numpy() * f, 1.0, 99.5)
        users = np.maximum(b.users.to_numpy() * f, 0)
        vol = b.vol.to_numpy() * f
        power = np.maximum(b.power_kw.to_numpy() * (1 + 0.5 * (f - 1)), 0.1)
        sun = np.clip(1 + 0.12 * np.sin(2 * np.pi * bucket / 9.0 + self.phase * 0.7) + 0.03 * rng.standard_normal(S), 0.0, 1.4)
        solar = b.solar_kw.to_numpy() * sun
        soc = np.clip(b.soc.to_numpy() + 1.2 * np.sin(2 * np.pi * bucket / 15.0 + self.phase) + 0.3 * rng.standard_normal(S), 0, 100)
        dl = b.dl.to_numpy() / (1 + 0.6 * (f - 1))
        ul = dl * 0.27
        lat = b.latency_ms.to_numpy() * (1 + 0.5 * (f - 1))
        drop = np.clip(b["drop"].to_numpy() * (1 + 2.5 * (f - 1)), 0.02, None)
        setup = np.clip(b.setup.to_numpy() - 4.0 * (f - 1), 80, 100)
        sinr = np.clip(b.sinr.to_numpy() - 5.0 * (f - 1), 0, 35)
        score = cx_score(drop, setup, dl, lat, sinr)
        cxl = cx_label(score); nl = net_label(prb, sinr)
        grid_up = b.grid_up.to_numpy(); src = b.source.astype(str).to_numpy()
        cab = b.cab_temp.to_numpy(); fuel = b.fuel.to_numpy(); gcl = b.grid_class.astype(str).to_numpy()
        tamper = b.tamper.to_numpy(); theft_n = b.theft_n.to_numpy(); cyber_n = b.cyber_n.to_numpy(); ctext = b.cyber_text.to_numpy()
        sites = []
        n_phys = 0; n_cyber = 0
        for k in range(S):
            phys = []
            if theft_n[k] > 0: phys.append("Possible fuel/battery theft detected")
            if tamper[k]: phys.append("Tamper sensor triggered")
            if cab[k] > 33: phys.append(f"Cabinet temperature {cab[k]:.0f} C")
            if fuel[k] < 20 and gcl[k] in ("poor", "low"): phys.append(f"Generator fuel low ({fuel[k]:.0f}%)")
            if (not grid_up[k]) and soc[k] < 25: phys.append(f"Grid down, battery {soc[k]:.0f}%")
            n_phys += bool(phys); n_cyber += bool(cyber_n[k] > 0)
            sites.append({
                "site_id": b.site_id.iat[k], "lat": float(b.lat.iat[k]), "lon": float(b.lon.iat[k]), "status": str(b.status.iat[k]), "tier": str(b.tier.iat[k]),
                "grid_class": gcl[k], "source": src[k], "grid_up": bool(grid_up[k]),
                "power_kw": round(float(power[k]), 2), "solar_kw": round(float(solar[k]), 2), "soc": round(float(soc[k]), 1), "soh": round(float(b.soh.iat[k]), 1),
                "fuel": round(float(fuel[k]), 1), "cab_temp": round(float(cab[k]), 1), "outage_risk_3h": b.outage_risk_3h.iat[k],
                "prb": round(float(prb[k]), 1), "users": round(float(users[k])), "vol": round(float(vol[k]), 2), "sinr": round(float(sinr[k]), 1), "rsrp": round(float(b.rsrp.iat[k]), 1),
                "dl": round(float(dl[k]), 1), "ul": round(float(ul[k]), 1), "latency_ms": round(float(lat[k]), 1), "setup": round(float(setup[k]), 2), "drop": round(float(drop[k]), 2),
                "net_status": str(nl[k]), "cx_score": float(score[k]), "cx_label": str(cxl[k]), "align": alignment_text(str(nl[k]), str(cxl[k])),
                "cyber_n": int(cyber_n[k]), "cyber_text": str(ctext[k]) if cyber_n[k] > 0 else "", "phys": phys, "cost24h": round(float(b.cost24h.iat[k]), 0)})
        # fleet-level aggregates computed from the same live numbers
        renew = float(np.minimum(solar, power).sum() / max(power.sum(), 1e-6) * 100)
        fleet = {"power_now_kw": float(power.sum()), "power_base_kw": float(b.power_kw.sum()), "solar_now_kw": float(solar.sum()), "renewable_now_pct": renew,
                 "avg_prb_pct": float(prb.mean()), "active_users": float(users.sum()), "cx_score": float(score.mean()), "call_drop_pct": float(drop.mean()),
                 "call_setup_pct": float(setup.mean()), "dl_mbps": float(dl.mean()), "latency_ms": float(lat.mean()), "sinr_db": float(sinr.mean()),
                 "sites_cx_good": int((cxl == "Good").sum()), "sites_cx_fair": int((cxl == "Fair").sum()), "sites_cx_poor": int((cxl == "Poor").sum()),
                 "sites_congested": int((nl == "Congested").sum()), "sites_with_cyber_24h": int(n_cyber), "sites_with_physical": int(n_phys)}
        # live energy mix: who is carrying the load right now
        solar_direct = np.minimum(solar, power); net = power - solar_direct
        gen_now = (~grid_up) & (src == "GENERATOR")
        parts = {"grid_kwh": float(np.where(grid_up, net, 0).sum()), "diesel_generator_kwh": float(np.where(gen_now, net, 0).sum()),
                 "solar_kwh": float(solar_direct.sum()), "battery_kwh": float(np.where((~grid_up) & ~gen_now, net, 0).sum())}
        tot = sum(parts.values()) or 1.0
        mix_now = {**{k: round(v, 1) for k, v in parts.items()}, "grid_pct": round(parts["grid_kwh"] / tot * 100, 1), "diesel_pct": round(parts["diesel_generator_kwh"] / tot * 100, 1),
                   "solar_pct": round(parts["solar_kwh"] / tot * 100, 1), "battery_pct": round(parts["battery_kwh"] / tot * 100, 1), "total_kwh": round(tot, 1), "unit": "kW (instantaneous)"}
        return sanitize({"bucket": bucket, "updated_ms": int(bucket * LIVE_PERIOD_S * 1000), "next_update_in_s": int(LIVE_PERIOD_S - (now % LIVE_PERIOD_S)),
                         "period_s": LIVE_PERIOD_S, "fleet": fleet, "mix_now": mix_now, "sites": sites})

    # ------------------------------------------------------------------ predicted fleet view (next 24 h)
    def _build_fleet_forecast(self, forecast, site_status):
        D = self.D
        out = {"available": False}
        try:
            if forecast is None or not len(forecast):
                return out
            fc = forecast.copy()

            def piv(metric):
                x = fc[fc.metric == metric].pivot_table(index="site_id", columns="h", values="forecast", aggfunc="mean")
                return x.reindex(D.sites).ffill(axis=0).fillna(0.0).to_numpy()
            power = piv("site_total_power_kw"); prb = piv("prb_utilization_pct"); vol = piv("data_volume_gb")
            hs = sorted(fc.h.unique())
            off = self.offset
            dl, drop, lat, setup, sinr = self._predict_cx_from_prb(prb, {k: v[:, None] for k, v in off.items()})
            sc = cx_score(drop, setup, dl, lat, sinr)
            ts = fc[fc.metric == "site_total_power_kw"].groupby("h").target_time.first().reindex(hs)
            risk = pd.to_numeric(site_status.outage_risk_3h, errors="coerce")
            out = {"available": True, "hours": [int(h) for h in hs], "timestamps": [str(t) for t in ts.values],
                   "power_mw": np.round(power.sum(axis=0) / 1000.0, 3).tolist(), "prb_pct": np.round(prb.mean(axis=0), 1).tolist(),
                   "data_tb": np.round(vol.sum(axis=0) / 1000.0, 2).tolist(), "cx_score": np.round(sc.mean(axis=0), 1).tolist(),
                   "call_drop_pct": np.round(drop.mean(axis=0), 2).tolist(), "dl_mbps": np.round(dl.mean(axis=0), 1).tolist(),
                   "sites_poor_cx_peak": int((sc.min(axis=1) < 60).sum()), "peak_prb_pct": round(float(prb.mean(axis=0).max()), 1),
                   "peak_prb_hour": int(hs[int(prb.mean(axis=0).argmax())]),
                   "avg_outage_risk_pct": round(float(risk.mean() * 100), 1) if risk.notna().any() else None,
                   "sites_outage_risk_high": int((risk >= 0.13).sum()), "grid_down_now": int(risk.isna().sum()),
                   "note": "Power, PRB and data volume are LightGBM forecasts; customer experience is derived from the forecast PRB using the fleet's measured load-vs-experience curves."}
        except Exception as e:  # never let the optional forecast panel stop the API from starting
            out = {"available": False, "error": str(e)}
        return sanitize(out)
