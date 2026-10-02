# Telecom AI Ops - AI Telco Operations Command Centre (v6) - Team Connected Minds

Topic: Intelligent Telecom: Energy, Network Security & ESG

## Run
python -m venv .venv
.venv\Scripts\activate                 (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cd api
uvicorn main:app --reload --port 8000
(then open http://127.0.0.1:8000 - start-up now takes a few extra seconds to build the cost / customer-experience tables)

## What's new in v6 (final improvements)
1. Every block on the Executive Dashboard and Site Explorer is labelled CURRENT VALUES, PREDICTED VALUES or MODEL DECISION
   (plus HISTORICAL for recorded trends), with a coloured section heading.
2. The "AI Network Sustainability & Security Orchestrator" capability list moved from the Executive Dashboard to the new About page.
3. Customer experience: fleet tiles + charts on the Executive Dashboard, current/24 h/predicted CX on each site, CX in the Trend view.
   CX score (0-100) = 30% call drop, 25% call set-up success, 20% downlink, 15% latency, 10% SINR (see About page).
4. About page: team name "Connected Minds" and the topic title.
5. Energy source mix now includes BATTERY (24 h view and a live "Now" view).
6. Live feed: /api/live changes every 2 minutes; the dashboard re-fetches automatically (KPIs, map, CX tiles, worst-sites table).
7. New Cost tab: overall, tariff, diesel, damages, ESG carbon, SLA penalties, lost revenue (unmet demand), cyber response; per site; 24 h / 7 days;
   AI-vs-baseline saving; assumptions list. Rates are in pipeline/s5_orchestrator.py CFG and api/insights.py - replace with operator rates.
8. Clicking "Command Centre / Network Sustainability & Security" opens the Executive Dashboard.
9. Hover (or click) a site on the map: energy, network performance, customer experience (aligned with network load) and cyber & physical alerts.

New backend file: api/insights.py. New endpoints: /api/live, /api/cx/fleet, /api/forecast/fleet, /api/costs/summary, /api/costs/sites.
/api/esg/mix now returns battery_kwh / battery_pct as well. /api/site/{id} now includes a "cx" block. No model or pipeline file was changed.

## Earlier: v5

Fixed two real gaps found by auditing the actual code (not just re-describing it):

1. Executive Dashboard KPI row now has 6 tiles instead of 5: added "Energy (24h)" in MWh with a
   day-over-day trend arrow, and "Est. monthly saving" in Rand (extrapolated from today's open
   energy-saving recommendations x the grid tariff x 30 - same assumption used in the per-site
   estimate, now also shown fleet-wide to match the brief's "Estimated monthly saving: R1.2m" style).
2. A "Cyber & physical alerts, last 7 days" panel now sits next to "Highest-priority items" on the
   Executive Dashboard. The backend endpoint (/api/cyber/alerts) existed since v1 but was never
   wired into the UI - it's a genuinely different, more useful view than the per-site Security
   recommendation, because it shows the alert-level history (885 alert-hours in the last 7 days,
   each with a time, site, severity and plain-language reason), not just "this site has an issue now".

No other screens changed. See the chat reply for the conceptual clarification on current status vs
forecast vs decision, which is worth re-reading before your demo.
