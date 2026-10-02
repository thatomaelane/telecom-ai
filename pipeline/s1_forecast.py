"""Stage 1: multi-horizon (1..24 h) LightGBM forecasts of data volume, PRB utilisation and site power.

Validation protocol (strict):
  train : 400 sites, days 8-40      valid : same sites, days 40-48 (early stopping)
  TEST-A: same sites, future days 48-60         (seen sites, unseen time)
  TEST-B: 100 HELD-OUT sites, days 48-60        (unseen sites AND unseen time)
Baselines: persistence, same-hour-yesterday, same-hour-last-week.
Then a final model per target is refit on all sites/all days for serving.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from numpy.lib.stride_tricks import sliding_window_view
from common import Data, metrics, save_json, MODELS, SERVING

TARGETS = {"data_volume_gb": "vol", "prb_utilization_pct": "prb", "site_total_power_kw": "power"}
STATIC_CAT = ["tier", "tech", "backhaul", "encl", "cat"]


def rolling(v, w, fn):
    out = np.full(v.shape, np.nan, np.float32)
    out[:, w - 1:] = fn(sliding_window_view(v, w, axis=1), axis=2)
    return out


class Feat:
    def __init__(self, D):
        self.D = D
        self.M = {c: D.a(c) for c in ["data_volume_gb", "prb_utilization_pct", "active_rrc_connected_users",
                                       "site_total_power_kw", "telecom_dc_load_kw", "ambient_outdoor_temp_c",
                                       "cabinet_internal_temp_c", "hvac_operational_status", "solar_generation_kw"]}
        self.roll = {}
        for c in TARGETS:
            v = self.M[c]
            self.roll[c] = dict(mean24=rolling(v, 24, np.mean), max24=rolling(v, 24, np.max),
                                min24=rolling(v, 24, np.min), mean168=rolling(v, 168, np.mean))

    def build(self, tgt, s, p, h):
        D, M, R = self.D, self.M, self.roll[tgt]
        v = M[tgt]
        th = p + h                                   # absolute index of target hour
        wk = th - 168
        f = {
            "h": h, "now": v[s, p], "lag1": v[s, p - 1], "lag2": v[s, p - 2], "lag3": v[s, p - 3], "lag6": v[s, p - 6],
            "mean24": R["mean24"][s, p], "max24": R["max24"][s, p], "min24": R["min24"][s, p], "mean168": R["mean168"][s, p],
            "same_h_yday": v[s, th - 24],
            "same_h_lastweek": np.where(wk >= 0, v[s, np.clip(wk, 0, None)], np.nan),
            "hour_t": D.hour[th], "dow_t": D.dow[th], "hol_t": D.hol[th], "weekend_t": (D.dow[th] >= 5).astype(np.int16),
        }
        for c in ["data_volume_gb", "prb_utilization_pct", "active_rrc_connected_users", "site_total_power_kw",
                  "ambient_outdoor_temp_c", "cabinet_internal_temp_c", "hvac_operational_status", "solar_generation_kw"]:
            if c != tgt:
                f["ctx_" + c[:8] + "_" + c[-4:]] = M[c][s, p]
        for c in ["tier", "tech", "backhaul", "encl", "cat", "lat", "lon"]:
            f[c] = D.static[c].values[s]
        return pd.DataFrame(f)


def sample_rows(rng, sites, p_lo, p_hi, T, frac=1.0):
    s, p = np.meshgrid(sites, np.arange(p_lo, min(p_hi, T - 25)), indexing="ij")
    s, p = s.ravel(), p.ravel()
    if frac < 1:
        k = rng.random(len(s)) < frac
        s, p = s[k], p[k]
    h = rng.integers(1, 25, len(s))
    return s, p, h


def main():
    t0 = time.time()
    D = Data(); F = Feat(D); T = D.T
    rng = np.random.default_rng(0)
    perm = rng.permutation(D.S)
    tr_sites, ho_sites = np.sort(perm[:400]), np.sort(perm[400:])
    np.save(MODELS / "holdout_sites.npy", ho_sites)
    split = {"train_sites": 400, "holdout_sites": 100}
    report = {"protocol": "train 400 sites d8-40 | valid d40-48 | TEST-A seen sites d48-60 | TEST-B 100 unseen sites d48-60"}
    for tgt, short in TARGETS.items():
        s_tr, p_tr, h_tr = sample_rows(rng, tr_sites, 168, 40 * 24, T, 0.5)
        s_va, p_va, h_va = sample_rows(rng, tr_sites, 40 * 24, 48 * 24, T, 0.3)
        Xtr, Xva = F.build(tgt, s_tr, p_tr, h_tr), F.build(tgt, s_va, p_va, h_va)
        ytr, yva = F.M[tgt][s_tr, p_tr + h_tr], F.M[tgt][s_va, p_va + h_va]
        params = dict(objective="regression", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1)
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=STATIC_CAT); dva = lgb.Dataset(Xva, yva, reference=dtr)
        mdl = lgb.train(params, dtr, 500, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
        best = mdl.best_iteration
        mdl.save_model(str(MODELS / f"forecast_{short}_eval.txt"), num_iteration=best)   # train-only model for honest backtests
        res = {"best_iteration": best}
        for name, sites in [("TEST_A_seen_sites_future", tr_sites), ("TEST_B_unseen_sites_future", ho_sites)]:
            s, p, h = sample_rows(rng, sites, 48 * 24, T, T, 0.35)
            X = F.build(tgt, s, p, h); y = F.M[tgt][s, p + h]
            pred = mdl.predict(X, num_iteration=best)
            r = {"model": metrics(y, pred),
                 "persistence": metrics(y, X["now"]),
                 "same_hour_yesterday": metrics(y, X["same_h_yday"]),
                 "same_hour_lastweek": metrics(y, X["same_h_lastweek"])}
            r["by_horizon_wMAPE"] = {f"{a}-{b}h": metrics(y[(h >= a) & (h <= b)], pred[(h >= a) & (h <= b)])["wMAPE"]
                                     for a, b in [(1, 3), (4, 12), (13, 24)]}
            res[name] = r
        imp = pd.Series(mdl.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
        res["top_features"] = (imp / imp.sum()).head(6).round(3).to_dict()
        report[tgt] = res
        # ---- final refit on all sites and all history for serving
        all_sites = np.arange(D.S)
        s_all, p_all, h_all = sample_rows(rng, all_sites, 168, T, T, 0.5)
        dall = lgb.Dataset(F.build(tgt, s_all, p_all, h_all), F.M[tgt][s_all, p_all + h_all], categorical_feature=STATIC_CAT)
        final = lgb.train(params, dall, int(best * 1.1))
        final.save_model(str(MODELS / f"forecast_{short}.txt"))
        print(f"{tgt}: best_iter={best} | A wMAPE model {res['TEST_A_seen_sites_future']['model']['wMAPE']:.1f}% vs yday "
              f"{res['TEST_A_seen_sites_future']['same_hour_yesterday']['wMAPE']:.1f}% | B model {res['TEST_B_unseen_sites_future']['model']['wMAPE']:.1f}% "
              f"vs yday {res['TEST_B_unseen_sites_future']['same_hour_yesterday']['wMAPE']:.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    save_json(report, MODELS / "report_forecast.json")

    # ---- next-24h forecasts from the last observed hour (for the dashboard)
    p = T - 1
    rows = []
    for tgt, short in TARGETS.items():
        mdl = lgb.Booster(model_file=str(MODELS / f"forecast_{short}.txt"))
        for h in range(1, 25):
            s = np.arange(D.S); pp = np.full(D.S, p); hh = np.full(D.S, h)
            pred = mdl.predict(F.build(tgt, s, pp, hh))
            rows.append(pd.DataFrame({"site_id": D.sites, "target_time": D.ts_ext[p + h], "h": h, "metric": tgt, "forecast": pred}))
    fc = pd.concat(rows)
    fc.to_parquet(SERVING / "forecast_next24h.parquet", index=False)
    print("saved forecasts", fc.shape)


if __name__ == "__main__":
    main()
