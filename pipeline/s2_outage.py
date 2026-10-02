"""Stage 2: grid-outage early warning. P(grid goes down at any point in the next 3 h | grid is up now).
Signals: voltage sag/brownout, recent outage history, 24h availability, hour of day, reliability class, regional stress.
Baseline: historical hazard by (reliability class, hour) learned on training sites/days only.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from common import Data, save_json, MODELS, SERVING


def main():
    D = Data(); S, T = D.S, D.T
    volt = D.a("grid_voltage_v"); down = volt == 0
    avail = D.a("grid_availability_hours_daily")
    onehot = np.zeros((D.region.max() + 1, S), np.float32); onehot[D.region, np.arange(S)] = 1
    reg_n = onehot.sum(1, keepdims=True)
    reg_frac = (onehot @ down.astype(np.float32)) / reg_n                     # (R,T)
    since = np.zeros((S, T), np.float32); cnt = np.full(S, 999.0)
    for t in range(T):
        cnt = np.where(down[:, t], 0, cnt + 1); since[:, t] = np.minimum(cnt, 999)
    fut = np.zeros((S, T), bool)
    for k in (1, 2, 3):
        fut[:, :T - k] |= down[:, k:]
    volt_pos = np.where(volt > 0, volt, np.nan)

    def feats(s, p):
        f = pd.DataFrame({
            "volt": volt[s, p], "volt_lag1": np.where(p > 0, volt[s, np.clip(p - 1, 0, None)], np.nan),
            "volt_min3": np.nanmin(np.stack([volt_pos[s, np.clip(p - k, 0, None)] for k in range(3)]), axis=0),
            "sag": 230 - volt[s, p], "avail24": avail[s, p], "since_outage_h": since[s, p],
            "region_frac_down": reg_frac[D.region[s], p], "hour": D.hour[p], "dow": D.dow[p],
            "cat": D.static.cat.values[s], "lat": D.static.lat.values[s], "lon": D.static.lon.values[s],
            "backhaul": D.static.backhaul.values[s], "tier": D.static.tier.values[s]})
        return f

    ho = np.load(MODELS / "holdout_sites.npy"); tr = np.setdiff1d(np.arange(S), ho)

    def rows(sites, lo, hi):
        s, p = np.meshgrid(sites, np.arange(lo, min(hi, T - 4)), indexing="ij"); s, p = s.ravel(), p.ravel()
        k = ~down[s, p]; return s[k], p[k]

    s_tr, p_tr = rows(tr, 24, 40 * 24); s_va, p_va = rows(tr, 40 * 24, 48 * 24)
    Xtr, Xva = feats(s_tr, p_tr), feats(s_va, p_va); ytr, yva = fut[s_tr, p_tr], fut[s_va, p_va]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=1)
    m = lgb.train(params, lgb.Dataset(Xtr, ytr, categorical_feature=["cat"]), 600,
                  valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    haz = pd.DataFrame({"cat": Xtr.cat, "hour": Xtr.hour, "y": ytr}).groupby(["cat", "hour"]).y.mean()
    rep = {"target": "any grid outage in next 1-3 h, given grid up now", "best_iteration": m.best_iteration}
    for name, sites in [("TEST_A_seen_sites_future", tr), ("TEST_B_unseen_sites_future", ho)]:
        s, p = rows(sites, 48 * 24, T); X = feats(s, p); y = fut[s, p]
        pr = m.predict(X, num_iteration=m.best_iteration)
        base = haz.reindex(pd.MultiIndex.from_arrays([X.cat, X.hour])).fillna(y.mean()).values
        prec, rec, thr = precision_recall_curve(y, pr)
        i = np.argmin(np.abs(rec - 0.5))
        rep[name] = {"base_rate": float(y.mean()), "AUC_model": float(roc_auc_score(y, pr)), "AP_model": float(average_precision_score(y, pr)),
                     "AUC_hazard_baseline": float(roc_auc_score(y, base)), "AP_hazard_baseline": float(average_precision_score(y, base)),
                     "precision_at_50pct_recall": float(prec[i]), "threshold_at_50pct_recall": float(thr[min(i, len(thr) - 1)])}
        print(name, {k: round(v, 3) for k, v in rep[name].items()}, flush=True)
    rep["top_features"] = (pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False) / m.feature_importance("gain").sum()).head(6).round(3).to_dict()
    # final refit on everything
    s_all, p_all = rows(np.arange(S), 24, T)
    fin = lgb.train(params, lgb.Dataset(feats(s_all, p_all), fut[s_all, p_all], categorical_feature=["cat"]), int(m.best_iteration * 1.1))
    fin.save_model(str(MODELS / "outage_risk.txt"))
    # current risk per site at last observed hour (for dashboard)
    s = np.arange(S); p = np.full(S, T - 1); X = feats(s, p)
    risk = pd.DataFrame({"site_id": D.sites, "grid_up_now": ~down[:, T - 1], "outage_risk_3h": fin.predict(X)})
    risk.loc[~risk.grid_up_now, "outage_risk_3h"] = np.nan
    risk.to_parquet(SERVING / "outage_risk_now.parquet", index=False)
    save_json(rep, MODELS / "report_outage.json")
    print("top features", rep["top_features"])


if __name__ == "__main__":
    main()
