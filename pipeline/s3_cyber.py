"""Stage 3: unsupervised cyber + physical anomaly detection (Isolation Forest on site-normalised signals).
Fit on days 0-40 (no labels used). Evaluate on days 40-60 against events_ground_truth.csv.
Baselines: (1) the site's own IDS flag, (2) simple z-score rule on logins/scans.
"""
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from common import Data, save_json, MODELS, SERVING, DATA

CYBER = ["brute_force", "port_scan", "traffic_flood", "modbus_intrusion", "regional_scan_campaign"]
PHYS = ["fuel_theft", "battery_cable_theft"]
FEATS = ["z_logins", "z_scans", "z_vol", "z_users", "z_lat", "tamper", "bms_fault", "ids_flag", "fuel_drop", "soc_drop"]


def mad_scale(x, axis):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return med, np.nanmedian(np.abs(x - med), axis=axis, keepdims=True) * 1.4826


def build(D, base_days=40):
    S, T = D.S, D.T; nb = base_days * 24
    L = np.log1p(D.a("failed_login_attempts_count")); C = np.log1p(D.a("unauthorized_ip_scan_count"))
    V = np.log(D.a("data_volume_gb")); U = np.log(D.a("active_rrc_connected_users")); LT = np.log(D.a("latency_rtt_ms"))
    z = {}
    for name, x in [("z_logins", L), ("z_scans", C)]:
        med, mad = mad_scale(x[:, :nb], 1); z[name] = (x - med) / np.maximum(mad, 0.35)
    for name, x in [("z_vol", V), ("z_users", U), ("z_lat", LT)]:
        xr = x[:, :nb].reshape(S, base_days, 24)
        med = np.median(xr, axis=1); mad = np.median(np.abs(xr - med[:, None, :]), axis=1) * 1.4826       # per site x hour-of-day
        z[name] = (x - np.tile(med, (1, T // 24))) / np.maximum(np.tile(mad, (1, T // 24)), 0.10)
    fuel = D.a("generator_fuel_level_pct"); genrun = D.a("generator_run_hours"); soc = D.a("battery_soc_pct")
    src_batt = (D.df.power_source_active.to_numpy().reshape(S, T) == "BATTERY")
    fd = np.zeros((S, T), np.float32); fd[:, 1:] = np.maximum(fuel[:, :-1] - fuel[:, 1:], 0) * (genrun[:, 1:] == 0)
    sd = np.zeros((S, T), np.float32); sd[:, 1:] = np.maximum(soc[:, :-1] - soc[:, 1:], 0) * (~src_batt[:, 1:])
    z["tamper"] = D.a("physical_tamper_sensor_status"); z["bms_fault"] = (D.df.bms_modbus_comm_status.to_numpy().reshape(S, T) == "FAULT").astype(np.float32)
    z["ids_flag"] = D.a("network_traffic_anomaly_flag"); z["fuel_drop"] = fd; z["soc_drop"] = sd
    return {k: v.astype(np.float32) for k, v in z.items()}


def labels(D):
    ev = pd.read_csv(DATA / "events_ground_truth.csv", parse_dates=["start", "end"])
    idx = {s: i for i, s in enumerate(D.sites)}
    lab = {"cyber": np.zeros((D.S, D.T), bool), "phys": np.zeros((D.S, D.T), bool)}
    e2 = ev[ev.site_id.isin(idx)]
    t0 = ((e2.start - D.ts[0]).dt.total_seconds() // 3600).astype(int).values; t1 = ((e2.end - D.ts[0]).dt.total_seconds() // 3600).astype(int).values
    for (_, r), a, b in zip(e2.iterrows(), t0, t1):
        g = "cyber" if r.event_type in CYBER else "phys" if r.event_type in PHYS else None
        if g: lab[g][idx[r.site_id], a:b + 1] = True
    return lab, e2.assign(t0=t0, t1=t1)


def main():
    D = Data(); Z = build(D); S, T = D.S, D.T
    X = np.stack([Z[f] for f in FEATS], axis=2)                                  # (S,T,F)
    rng = np.random.default_rng(1)
    tr_idx = np.argwhere(np.ones((S, 40 * 24), bool))
    tr_idx = tr_idx[rng.choice(len(tr_idx), 250000, replace=False)]
    iso = IsolationForest(n_estimators=250, max_samples=4096, contamination="auto", random_state=0, n_jobs=1)
    iso.fit(X[tr_idx[:, 0], tr_idx[:, 1]])
    te = slice(40 * 24, T)
    score = -iso.score_samples(X[:, te].reshape(-1, len(FEATS))).reshape(S, -1)
    lab, ev = labels(D)
    rule = np.maximum(Z["z_logins"], Z["z_scans"])[:, te]
    ids = Z["ids_flag"][:, te]
    rep = {"fit_window": "days 0-40 unsupervised", "eval_window": "days 40-60"}
    for grp, cols in [("cyber_incidents", "cyber"), ("physical_theft", "phys")]:
        y = lab[cols][:, te].ravel()
        r = {"positive_hours": int(y.sum()), "base_rate": float(y.mean())}
        for nm, sc in [("isolation_forest", score), ("zscore_rule", rule), ("site_IDS_flag", ids)]:
            s = sc.ravel(); r[nm] = {"AUC": float(roc_auc_score(y, s)), "AP": float(average_precision_score(y, s))}
        rep[grp] = r
    # ---- two-lane alerting: Lane A = site-normalised login/scan statistics (volumetric attacks),
    #      Lane B = Isolation Forest (multivariate: traffic flood, Modbus/OT), Lane C = physical-theft rule.  ~1% alert budget.
    thr_rule = float(np.quantile(rule, 0.993)); thr_if = float(np.quantile(score, 0.997))
    laneA, laneB = rule >= thr_rule, score >= thr_if
    laneC = ((Z["fuel_drop"] > 10) | ((Z["tamper"] > 0) & (Z["soc_drop"] > 20)))[:, te]   # physical rule: fuel falling with genset off, or tamper + sudden SOC loss
    al = laneA | laneB | laneC
    yc, yp = lab["cyber"][:, te], lab["phys"][:, te]
    rep["two_lane_alerting"] = {"thr_rule_z": thr_rule, "thr_isoforest": thr_if, "alerts_per_day_fleet": float(al.sum() / 20),
        "alert_share_of_site_hours_pct": float(al.mean() * 100), "cyber_hour_precision": float(yc[al].mean()),
        "cyber_hour_recall": float(al[yc].mean()), "phys_hour_recall": float(al[yp].mean()),
        "baseline_site_IDS_precision": float(yc[ids > 0].mean()), "baseline_site_IDS_recall": float((ids > 0)[yc].mean()),
        "baseline_site_IDS_alerts_per_day_fleet": float((ids > 0).sum() / 20)}
    det = {}
    for typ, g in ev[ev.t1 >= 40 * 24].groupby("event_type"):
        if typ not in CYBER + PHYS: continue
        hits = []
        for _, r in g.iterrows():
            a, b = max(r.t0, 40 * 24) - 40 * 24, r.t1 - 40 * 24
            hits.append(bool(al[np.where(D.sites == r.site_id)[0][0], a:b + 3].any()))
        det[typ] = {"events": len(hits), "detected_pct": round(100 * float(np.mean(hits)), 1)}
    rep["event_level_detection_two_lane"] = det
    joblib.dump({"model": iso, "features": FEATS, "thr_rule_z": thr_rule, "thr_if": thr_if, "baseline_days": 40}, MODELS / "cyber_isoforest.joblib")
    save_json(rep, MODELS / "report_cyber.json")
    for k in ("cyber_incidents", "physical_theft", "two_lane_alerting", "event_level_detection_two_lane"):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in rep[k].items()})
    # ---- serving: alerts for the most recent 7 days with a plain-language reason
    n7 = 7 * 24; sc7, ru7, al7 = score[:, -n7:], rule[:, -n7:], al[:, -n7:]
    ss, tt = np.where(al7); rows = []
    for s_, t_ in zip(ss, tt):
        tg = T - n7 + t_; fz = {f: Z[f][s_, tg] for f in FEATS}
        if fz["fuel_drop"] > 10 or (fz["tamper"] and fz["soc_drop"] > 20): why, sev = "Possible fuel/battery theft (tamper + sudden drop)", "critical"
        elif fz["z_vol"] > 4 or fz["z_users"] > 4: why, sev = "Traffic flood / DDoS-like surge", "high"
        elif fz["z_scans"] > 4 and fz["z_scans"] >= fz["z_logins"]: why, sev = "Port-scan / reconnaissance burst", "high" if fz["z_scans"] > 8 else "medium"
        elif fz["z_logins"] > 4: why, sev = "Brute-force login attempts", "high" if fz["z_logins"] > 8 else "medium"
        elif fz["bms_fault"]: why, sev = "BMS/Modbus communication fault", "medium"
        else: why, sev = "Combined behavioural anomaly", "medium"
        rows.append((D.sites[s_], D.ts[tg], float(sc7[s_, t_]), float(ru7[s_, t_]), why, sev))
    out = pd.DataFrame(rows, columns=["site_id", "time", "if_score", "z_login_scan", "reason", "severity"])
    out.sort_values("time", ascending=False).to_parquet(SERVING / "cyber_alerts_7d.parquet", index=False)
    print("alerts last 7d:", len(out), out.severity.value_counts().to_dict())


if __name__ == "__main__":
    main()
