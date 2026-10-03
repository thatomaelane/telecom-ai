"""FastAPI backend for the AI Telco Operations Command Centre.

Run:  uvicorn main:app --reload --port 8000     (from this folder)
Docs: http://127.0.0.1:8000/docs                 (auto-generated, try every endpoint here first)
Site query engine (s6_query.Engine) is loaded ONCE at startup, not per request - see lifespan below.
"""
import sys, json
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np, pandas as pd
import io, time
from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# From the team's Description_of_the_Dataset.pdf - the 7 telemetry groups + static site characteristics
COLUMN_GROUPS = {
    "Network demand": ["data_volume_gb", "prb_utilization_pct", "active_rrc_connected_users", "busy_hour_traffic_ratio"],
    "Energy": ["site_total_power_kw", "site_total_energy_kwh", "telecom_dc_load_kw", "site_pue", "grid_voltage_v"],
    "Power sources": ["power_source_active", "grid_availability_hours_daily", "battery_soc_pct", "battery_soh_pct",
                       "solar_generation_kw", "generator_fuel_level_pct", "generator_run_hours"],
    "Temperature & cooling": ["cabinet_internal_temp_c", "battery_bank_temp_c", "ambient_outdoor_temp_c",
                               "hvac_operational_status", "cooling_fan_speed_rpm"],
    "Cybersecurity": ["failed_login_attempts_count", "network_traffic_anomaly_flag", "physical_tamper_sensor_status",
                       "unauthorized_ip_scan_count", "bms_modbus_comm_status"],
    "ESG": ["diesel_consumed_liters", "scope_1_carbon_kg_co2e", "scope_2_carbon_kg_co2e",
            "renewable_energy_share_pct", "battery_recycling_status"],
    "Customer experience": ["call_setup_success_rate_pct", "call_drop_rate_pct", "avg_dl_throughput_mbps",
                             "avg_ul_throughput_mbps", "latency_rtt_ms", "rsrp_dbm", "sinr_db"],
    "Site characteristics": ["site_demographic_tier", "tower_structure_type", "enclosure_type", "backhaul_technology",
                              "deployed_technologies", "grid_reliability_category", "latitude", "longitude"],
}
REQUIRED_MIN_COLUMNS = ["site_id", "timestamp", "site_total_power_kw", "data_volume_gb"]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s6_query import Engine          # noqa: E402
from common import SERVING, MODELS   # noqa: E402
from insights import Insights, sanitize, LIVE_PERIOD_S   # noqa: E402  (CX, cost, live feed, fleet forecast)

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and data (about 10-20s)...")
    STATE["engine"] = Engine()
    STATE["kpis"] = json.loads((SERVING / "kpis.json").read_text())
    STATE["site_status"] = pd.read_parquet(SERVING / "site_status_latest.parquet")
    STATE["reco"] = pd.read_parquet(SERVING / "recommendations_now.parquet")
    STATE["esg_daily"] = pd.read_parquet(SERVING / "esg_daily.parquet")
    STATE["esg_class"] = pd.read_parquet(SERVING / "esg_by_class_daily.parquet")
    STATE["cyber"] = pd.read_parquet(SERVING / "cyber_alerts_7d.parquet")
    STATE["orch_sites"] = pd.read_parquet(SERVING / "orchestrator_sites.parquet")
    STATE["orch_report"] = json.loads((MODELS / "report_orchestrator.json").read_text())
    STATE["fc_report"] = json.loads((MODELS / "report_forecast.json").read_text())
    try:
        fc = pd.read_parquet(SERVING / "forecast_next24h.parquet")
    except Exception as e:
        print("forecast_next24h.parquet not loaded:", e); fc = None
    print("Building customer-experience, cost and live-feed tables...")
    STATE["ins"] = Insights(STATE["engine"], STATE["site_status"], STATE["cyber"], fc)
    import gc; gc.collect()
    try:                                  # Linux only: return freed start-up memory to the OS (keeps small hosts under their limit)
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    print("Ready.")
    yield
    STATE.clear()


app = FastAPI(title="AI Telco Operations Command Centre", version="0.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def df_json(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ------------------------------------------------------------------ Executive Dashboard
@app.get("/api/kpis")
def kpis():
    """Fleet-wide headline numbers for the Executive Dashboard tiles."""
    return STATE["kpis"]


@app.get("/api/sites")
def sites(status: str | None = None):
    """All 500 sites with map coordinates + status, for the network map / site table.
    Filter with ?status=critical|alert|backup|normal"""
    df = STATE["site_status"]
    if status:
        df = df[df.status == status]
    return df_json(df)


@app.get("/api/esg/daily")
def esg_daily():
    """60-day ESG trend (energy, Scope 1+2, diesel, renewable share) for the ESG chart."""
    return df_json(STATE["esg_daily"])


@app.get("/api/esg/by_class")
def esg_by_class():
    """Same, split by grid-reliability class (poor/low/medium/high)."""
    return df_json(STATE["esg_class"])


@app.get("/api/esg/mix")
def esg_mix(hours: int = 24):
    """Fleet energy-source mix over the last N hours: grid, diesel generator, solar PV and BATTERY.
    These are the sources that actually carried the site load. Battery = stored energy discharged while the
    grid was down and the generator was off (it was charged earlier from grid / solar / diesel, so it is a
    carrier, not a primary source). Legacy keys (grid_kwh, diesel_generator_kwh, solar_kwh, *_pct) are kept."""
    return STATE["ins"].mix(hours)


@app.get("/api/live")
def live():
    """The 'live' Executive Dashboard feed. The latest recorded hour for every site, with a smooth simulated
    variation that changes every 2 minutes (load-driven values and customer experience move together).
    Fleet KPIs, the map, the energy mix and the fleet customer-experience tiles are all computed from this."""
    ins = STATE["ins"]; out = ins.live(); k = STATE["kpis"]
    fl = out["fleet"]; ratio = fl["power_now_kw"] / max(fl["power_base_kw"], 1e-6)
    damp = 1 + 0.25 * (ratio - 1)       # the rolling 24 h figures move slowly relative to the instantaneous load
    out["kpis"] = {"as_of": k["as_of"], "sites": k["sites"], "status_counts": k["status_counts"],
                   "energy_24h_kwh": k["energy_last24h_kwh"] * damp, "co2e_24h_kg": k["co2e_last24h_kg"] * damp,
                   "renewable_24h_pct": k["renewable_last24h_pct"], "cyber_alerts_24h": k["cyber_alerts_24h"],
                   "potential_saving_kwh_24h": k["potential_saving_kwh_24h"], "potential_saving_kg_24h": k["potential_saving_kg_24h"],
                   "grid_down_sites": k["grid_down_sites"], "recs": k.get("recs", {})}
    return sanitize(out)


@app.get("/api/cx/fleet")
def cx_fleet(hours: int = 72):
    """Fleet customer-experience time series (call drop, set-up success, throughput, latency, SINR) next to the
    network load (PRB %) so the two can be read together. Recorded values."""
    return STATE["ins"].cx_fleet(hours)


@app.get("/api/forecast/fleet")
def forecast_fleet():
    """PREDICTED VALUES for the Executive Dashboard: next 24 h fleet power, PRB, data volume, derived customer
    experience, and the grid-outage risk view."""
    return STATE["ins"].forecast_cache


@app.get("/api/costs/summary")
def costs_summary(hours: int = 24):
    """Cost tab: overall cost, tariff, diesel, damages, ESG carbon, SLA penalties, lost revenue and cyber response
    for the last 24 h or 7 days, with a time series, a grid-class split, AI-vs-baseline saving and the assumptions."""
    return STATE["ins"].cost_summary(hours)


@app.get("/api/costs/sites")
def costs_sites(hours: int = 24, sort: str = "total", q: str | None = None, limit: int = 50):
    """Cost per site (sortable, searchable) for the same window."""
    return STATE["ins"].cost_sites(hours, sort, q, limit)


# ------------------------------------------------------------------ AI Insights / Recommendations
@app.get("/api/recommendations")
def recommendations(category: str | None = None, priority: str | None = None, site_id: str | None = None):
    """Fleet-wide recommendation queue (energy saving, resilience, fuel, battery health, security, thermal)."""
    df = STATE["reco"]
    if category: df = df[df.category == category]
    if priority: df = df[df.priority == priority]
    if site_id: df = df[df.site_id == site_id]
    return df_json(df.sort_values("priority", ascending=False))


@app.get("/api/cyber/alerts")
def cyber_alerts(severity: str | None = None):
    """Last 7 days of cyber/physical alerts with plain-language reasons."""
    df = STATE["cyber"]
    if severity: df = df[df.severity == severity]
    return df_json(df)


@app.get("/api/model_performance")
def model_performance():
    """Backtest metrics (forecast accuracy, outage/cyber detection) - for an 'AI trust' panel."""
    f = STATE["fc_report"]
    return {"forecast": {k: f[k]["TEST_B_unseen_sites_future"]["model"] for k in ["data_volume_gb", "prb_utilization_pct", "site_total_power_kw"]}}


# ------------------------------------------------------------------ Site explorer (the core query)
@app.get("/api/site/{site_id}")
def site_snapshot(site_id: str, when: str = Query(..., description="e.g. 2026-09-15 03:00"), horizon_h: int = 24):
    """THE core dashboard query: pick a site + date/time -> current status, next-h predictions,
    financial impact, ESG, cybersecurity, maintenance outlook, and a smart decision."""
    if site_id not in STATE["engine"].site_idx:
        raise HTTPException(404, f"Unknown site_id. Valid range: SITE_001 .. SITE_{STATE['engine'].D.S:03d}")
    try:
        eng = STATE["engine"]
        r = eng.snapshot(site_id, when, horizon_h=min(horizon_h, 24))
        i, p = eng._p(site_id, when)
        r["cx"] = STATE["ins"].site_cx(i, p, r["predictions_next_h"]["prb_utilization_pct"])   # current + 24 h history + predicted CX + 24 h cost
        return r
    except Exception as e:
        raise HTTPException(400, f"Could not build snapshot: {e}")


@app.get("/api/site/{site_id}/history")
def site_history(site_id: str, start: str = Query(...), end: str = Query(...)):
    """Date-range trend for the Site Explorer 'Trend' view - actual recorded telemetry between two dates
    (not a forecast), plus summary stats (avg/peak power, total CO2e, total cost, downtime hours)."""
    if site_id not in STATE["engine"].site_idx:
        raise HTTPException(404, f"Unknown site_id. Valid range: SITE_001 .. SITE_{STATE['engine'].D.S:03d}")
    try:
        eng = STATE["engine"]
        r = eng.history(site_id, start, end)
        i, p0 = eng._p(site_id, start); _, p1 = eng._p(site_id, end)
        p0, p1 = min(p0, p1), max(p0, p1)
        if p1 - p0 > 24 * 92:
            p1 = p0 + 24 * 92
        extra = STATE["ins"].site_history_extra(i, p0, p1)
        if all(len(v) == len(r["series"]) for v in extra.values()):
            for k, row in enumerate(r["series"]):
                for name, vals in extra.items():
                    row[name] = vals[k]
        return r
    except Exception as e:
        raise HTTPException(400, f"Could not build history: {e}")


@app.post("/api/simulate")
def simulate(payload: dict = Body(...)):
    """The 'Simulate' live-telemetry mode: an operator on site enters current readings by hand (no
    historical lookup) and gets an immediate cost/ESG/cyber read and a decision. Required: power_kw,
    prb_utilization_pct. Everything else has a sensible default - see Engine.instant_assessment."""
    try:
        return STATE["engine"].instant_assessment(payload)
    except KeyError as e:
        raise HTTPException(400, f"Missing required field: {e}")
    except Exception as e:
        raise HTTPException(400, f"Could not run assessment: {e}")


@app.get("/api/maintenance")
def maintenance_queue(urgency: str | None = None):
    """Fleet-wide predictive-maintenance queue: battery/generator/cooling items with an ETA and an
    estimated avoidable cost, derived from each site's 60-day trend."""
    eng = STATE["engine"]; rows = []
    for site_id in eng.D.sites:
        for m in eng.maintenance_for(site_id):
            rows.append({"site_id": site_id, **m})
    df = pd.DataFrame(rows)
    if urgency and len(df): df = df[df.urgency == urgency]
    order = {"high": 0, "medium": 1, "low": 2}
    if len(df): df = df.assign(_o=df.urgency.map(order)).sort_values("_o").drop(columns="_o")
    return df_json(df) if len(df) else []


@app.get("/api/meta")
def meta():
    """Valid site ids and the dataset's date range, for the dashboard's dropdown/date-picker."""
    D = STATE["engine"].D
    return {"sites": D.sites.tolist(), "date_min": str(D.ts[0]), "date_max": str(D.ts[D.T - 1]), "hours": D.T}


# ------------------------------------------------------------------ What-if simulator
@app.get("/api/whatif/baseline_vs_smart")
def whatif_baseline_vs_smart():
    """The two full-fleet policies (reactive baseline vs smart orchestrator) already simulated over 60 days -
    the headline numbers for the What-if screen's default view."""
    r = STATE["orch_report"]
    return {"baseline": r["baseline"], "smart": r["smart"], "delta": r["delta"], "attribution": r["attribution"], "config": r["config"]}


@app.get("/api/whatif/sites")
def whatif_sites(decision: str | None = None):
    """Per-site energy-saving % vs availability-risk classification vs recommendation -
    the exact 'JHB-047 / CPT-112' style table from the brief."""
    df = STATE["orch_sites"]
    if decision: df = df[df.decision == decision]
    return df_json(df)


@app.get("/api/dataset/info")
def dataset_info():
    """Schema, size and date range of the active dataset, grouped the way the team's data dictionary
    groups it (Network demand / Energy / Power sources / Temperature & cooling / Cybersecurity / ESG /
    Customer experience / Site characteristics) - for a Dataset page a judge or operator can browse."""
    D = STATE["engine"].D
    groups = [{"group": g, "columns": [c for c in cols if c in D.df.columns]} for g, cols in COLUMN_GROUPS.items()]
    return {"rows": int(len(D.df)), "sites": int(D.S), "hours_per_site": int(D.T),
            "date_min": str(D.ts[0]), "date_max": str(D.ts[D.T - 1]), "total_columns": int(D.df.shape[1]),
            "groups": groups, "source_file": "telecom_synthetic_60d_500sites.parquet (generated from the team's original 1-day/500-site dataset)"}


@app.get("/api/dataset/sample")
def dataset_sample(site_id: str | None = None, limit: int = 50, offset: int = 0):
    """Browsable raw rows for the Dataset page - optionally filtered to one site."""
    D = STATE["engine"].D
    df = D.df if not site_id else D.df[D.df.site_id == site_id]
    limit = min(limit, 500)
    return {"total_rows": int(len(df)), "columns": list(df.columns), "rows": df_json(df.iloc[offset:offset + limit])}


@app.post("/api/dataset/upload")
async def dataset_upload(file: UploadFile = File(...)):
    """Lets an operator upload their own site telemetry (CSV or XLSX) to check it against this system's
    schema and preview it. This does NOT retrain the models live (that takes several minutes offline -
    see the pipeline README) - it validates the file's columns/types and returns a preview so an operator
    can confirm their export is compatible before someone runs the pipeline on it."""
    name = file.filename or "upload"
    raw = await file.read()
    if len(raw) > 60 * 1024 * 1024:
        raise HTTPException(400, "File too large (60 MB limit for this demo).")
    try:
        if name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file as CSV/XLSX: {e}")

    all_known = sum(COLUMN_GROUPS.values(), [])
    present = [c for c in df.columns if c in all_known]
    missing_required = [c for c in REQUIRED_MIN_COLUMNS if c not in df.columns]
    saved_path = None
    if not missing_required:
        saved_path = UPLOAD_DIR / f"{int(time.time())}_{Path(name).name}"
        raw_ext = ".xlsx" if name.lower().endswith((".xlsx", ".xls")) else ".csv"
        (saved_path.with_suffix(raw_ext)).write_bytes(raw)

    return {
        "filename": name, "rows": int(len(df)), "columns_found": int(df.shape[1]),
        "recognised_columns": len(present), "recognised_of_known": f"{len(present)}/{len(all_known)}",
        "missing_required_columns": missing_required,
        "valid": len(missing_required) == 0,
        "message": "File accepted and saved. To use it for predictions, run the pipeline (see README) with TELECOM_DATA pointed at this file's folder." if not missing_required
                   else f"Missing required columns: {', '.join(missing_required)}. This file cannot be used as-is.",
        "preview": df_json(df.head(20)),
    }


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent / "static"), html=True), name="static")
