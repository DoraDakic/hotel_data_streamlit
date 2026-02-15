import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# Configuration (mirrors CHP_Analysis_FIXED.ipynb)
# ============================================================

DEFAULT_ANALYSIS_YEAR = 2016

DT_HOURS = 0.5  # 30-min data
BASELOAD_TIME_START = "00:00"
BASELOAD_TIME_END = "06:00"
BASELOAD_QUANTILE = 0.15

# Finance assumptions (can be overridden in sidebar)
DEFAULT_ELEC_PRICE = 0.30
DEFAULT_GAS_PRICE = 0.06
DEFAULT_DISCOUNT_RATE = 0.06
DEFAULT_ESCALATION = 0.02
BASELINE_BOILER_EFF = 0.75
NEW_BOILER_EFF = 0.95
DEFAULT_CHP_CAPEX_PER_KWE = 2500.0

# ============================================================
# Core methods
# ============================================================

def transpose_moodle_electric_csv(
    src: Path,
    start: str = "2015-01-01 00:30:00",
    end: str = "2017-01-01 00:00:00",
    freq: str = "30min",
) -> pd.DataFrame:
    """Transpose Moodle-format electrical demand file into tidy time series."""
    raw = pd.read_csv(src, header=None)
    raw = raw.drop(0, axis=1).drop(0, axis=0)

    dt_index = pd.date_range(start, end, freq=freq)
    out = pd.DataFrame({"date": dt_index})
    out["power_kW"] = pd.to_numeric(raw.stack().reset_index(drop=True), errors="coerce")
    return out


def load_power_timeseries(df: pd.DataFrame) -> pd.Series:
    """Return half-hourly power series (kW) indexed by datetime."""
    d = df.copy()

    # Handle common column names
    if "power_kW" not in d.columns:
        for c in ["power", "Load (kW)", "load_kW", "kW", "demand_kW"]:
            if c in d.columns:
                d = d.rename(columns={c: "power_kW"})
                break

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce", dayfirst=True)
        d = d.dropna(subset=["date"]).set_index("date")
    elif not isinstance(d.index, pd.DatetimeIndex):
        # Try first column as datetime
        d.iloc[:, 0] = pd.to_datetime(d.iloc[:, 0], errors="coerce", dayfirst=True)
        d = d.set_index(d.columns[0])

    d = d.sort_index()
    s = pd.to_numeric(d["power_kW"], errors="coerce")
    s.name = "power_kW"
    return s


def clean_power_series(power_kW: pd.Series, max_gap_steps: int = 12) -> pd.Series:
    """Notebook-consistent cleaning:
    - <=0 treated as missing
    - time interpolation capped at max_gap_steps (30-min steps)
    """
    s = power_kW.astype(float).copy()
    s[s <= 0] = np.nan
    s = s.interpolate(method="time", limit=max_gap_steps, limit_direction="both")
    return s


def estimate_baseload_kW(power_kW: pd.Series, year: int) -> float:
    s = power_kW.loc[str(year)].copy()
    night = s.between_time(BASELOAD_TIME_START, BASELOAD_TIME_END)
    return float(np.nanquantile(night.to_numpy(), BASELOAD_QUANTILE))


def validated_annual_kwh(power_kW: pd.Series, year: int) -> float:
    """Validated notebook method: rolling pairs (2x30min) * 0.5 -> hourly kWh."""
    s = power_kW.loc[str(year)].copy()
    df_kwh = s.to_frame(name="power_kW").rolling(window=2).sum() * 0.5
    df_kwh = df_kwh.iloc[1::2]
    return float(df_kwh["power_kW"].sum())


@dataclass(frozen=True)
class CHPSimulationResult:
    year: int
    chp_kWe: float
    baseload_kW: float
    peak_site_kW: float
    site_demand_kWh: float
    chp_gen_kWh: float
    self_consumed_chp_kWh: float
    export_kWh: float
    grid_import_kWh: float


def simulate_constant_chp_against_demand(demand_kW: pd.Series, chp_kWe: float, year: int) -> CHPSimulationResult:
    """Always-on CHP (electric) against site demand, timestep-consistent."""
    d = demand_kW.loc[str(year)].astype(float)

    demand_kWh_ts = d * DT_HOURS
    chp_kWh_ts = pd.Series(float(chp_kWe), index=d.index) * DT_HOURS

    self_use = np.minimum(demand_kWh_ts.to_numpy(), chp_kWh_ts.to_numpy())
    export = np.maximum(chp_kWh_ts.to_numpy() - demand_kWh_ts.to_numpy(), 0.0)
    grid_import = np.maximum(demand_kWh_ts.to_numpy() - chp_kWh_ts.to_numpy(), 0.0)

    # IMPORTANT: site demand kWh should use validated method (matches notebook)
    site_demand_kWh = validated_annual_kwh(demand_kW, year=year)
    chp_gen_kWh = float(np.nansum(chp_kWh_ts.to_numpy()))

    return CHPSimulationResult(
        year=int(year),
        chp_kWe=float(chp_kWe),
        baseload_kW=float(estimate_baseload_kW(demand_kW, year=year)),
        peak_site_kW=float(np.nanmax(d.to_numpy())),
        site_demand_kWh=float(site_demand_kWh),
        chp_gen_kWh=float(chp_gen_kWh),
        self_consumed_chp_kWh=float(np.nansum(self_use)),
        export_kWh=float(np.nansum(export)),
        grid_import_kWh=float(np.nansum(grid_import)),
    )


# -----------------------------
# Heat profile (monthly gas + sleepers -> 30-min heat demand)
# -----------------------------

def _days_in_month(ts: pd.Timestamp) -> int:
    return (ts + pd.offsets.MonthEnd(1)).day


def tidy_monthly_gas_occ(xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rea Original Excel layout: Monthly Gas Consumption + Hotel Occupancy."""
    gas_raw = pd.read_excel(xlsx, sheet_name="Monthly Gas Consumption")
    gas_monthly = (
        gas_raw[["Unnamed: 1", "Unnamed: 3"]]
        .rename(columns={"Unnamed: 1": "month", "Unnamed: 3": "gas_fuel_kWh"})
        .dropna(subset=["month", "gas_fuel_kWh"])
    )
    gas_monthly["month"] = pd.to_datetime(gas_monthly["month"]).dt.to_period("M").dt.to_timestamp()
    gas_monthly["gas_fuel_kWh"] = pd.to_numeric(gas_monthly["gas_fuel_kWh"], errors="coerce")
    gas_monthly = gas_monthly.dropna(subset=["gas_fuel_kWh"]).sort_values("month").reset_index(drop=True)

    occ_raw = pd.read_excel(xlsx, sheet_name="Hotel Occupancy")
    occ_block = (
        occ_raw.loc[2:13, ["Unnamed: 1", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]]
        .rename(columns={"Unnamed: 1": "month_name", "Unnamed: 2": 2015, "Unnamed: 3": 2016, "Unnamed: 4": 2017})
    )
    month_map = {m: i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"],
        start=1
    )}
    occ_long = occ_block.melt(id_vars=["month_name"], var_name="year", value_name="sleepers")
    occ_long["year"] = occ_long["year"].astype(int)
    occ_long["sleepers"] = pd.to_numeric(occ_long["sleepers"], errors="coerce")
    occ_long["month"] = occ_long["month_name"].map(month_map)
    occ_long["month"] = pd.to_numeric(occ_long["month"], errors="coerce")
    occ_long = occ_long.dropna(subset=["sleepers", "month"])
    occ_long["date"] = pd.to_datetime(dict(year=occ_long["year"], month=occ_long["month"].astype(int), day=1))
    occ_monthly = occ_long[["date", "sleepers"]].rename(columns={"date": "month"}).sort_values("month").reset_index(drop=True)

    return gas_monthly, occ_monthly


def build_heat_profile_from_monthly(
    gas_monthly: pd.DataFrame,
    occ_monthly: pd.DataFrame,
    year: int,
    boiler_eff: float = BASELINE_BOILER_EFF,
) -> pd.DataFrame:
    """Return 30-min profile with columns: heat_output_kW (useful) and gas_input_kW (fuel)."""
    g = gas_monthly[gas_monthly["month"].dt.year == year].copy()
    o = occ_monthly[occ_monthly["month"].dt.year == year].copy()
    m = g.merge(o, on="month", how="left")
    if m["sleepers"].isna().any():
        m["sleepers"] = m["sleepers"].fillna(m["sleepers"].mean())

    # DHW energy per sleeper-month from summer months (Jun–Aug)
    summer = m[m["month"].dt.month.isin([6, 7, 8])].copy()
    if len(summer) >= 2 and summer["sleepers"].gt(0).all():
        dhw_kWh_per_sleeper_month = float((summer["gas_fuel_kWh"] * boiler_eff / summer["sleepers"]).median())
    else:
        dhw_kWh_per_sleeper_month = 25.0

    m["dhw_useful_kWh"] = dhw_kWh_per_sleeper_month * m["sleepers"]
    m["total_useful_kWh"] = m["gas_fuel_kWh"] * boiler_eff
    m["dhw_useful_kWh"] = np.minimum(m["dhw_useful_kWh"], 0.8 * m["total_useful_kWh"])
    m["space_useful_kWh"] = m["total_useful_kWh"] - m["dhw_useful_kWh"]

    # 24h shapes (weekday/weekend)
    dhw_wd_24 = np.array([
        0.4,0.3,0.3,0.3,0.4,0.7,1.6,2.0,1.6,1.0,0.7,0.5,
        0.4,0.4,0.4,0.5,0.8,1.2,1.5,1.4,1.1,0.8,0.6,0.5
    ], dtype=float)
    dhw_we_24 = np.array([
        0.4,0.3,0.3,0.3,0.4,0.6,1.0,1.4,1.7,1.5,1.1,0.9,
        0.8,0.8,0.8,0.9,1.1,1.3,1.4,1.3,1.1,0.9,0.7,0.6
    ], dtype=float)
    space_wd_24 = np.array([
        1.2,1.15,1.1,1.05,1.0,1.1,1.25,1.3,1.15,1.0,0.9,0.85,
        0.8,0.8,0.85,0.95,1.05,1.1,1.1,1.05,1.0,0.95,0.9,0.9
    ], dtype=float)
    space_we_24 = np.array([
        1.2,1.15,1.1,1.05,1.0,1.05,1.15,1.2,1.15,1.05,0.95,0.9,
        0.85,0.85,0.9,0.95,1.0,1.05,1.05,1.0,0.95,0.9,0.9,0.9
    ], dtype=float)

    def to_halfhour(shape24: np.ndarray) -> np.ndarray:
        shape24 = shape24 / shape24.sum()
        return np.repeat(shape24, 2)  # 48 half-hours

    dhw_wd_48 = to_halfhour(dhw_wd_24)
    dhw_we_48 = to_halfhour(dhw_we_24)
    space_wd_48 = to_halfhour(space_wd_24)
    space_we_48 = to_halfhour(space_we_24)

    parts = []
    for _, row in m.iterrows():
        month = row["month"]
        start = month
        end = month + pd.offsets.MonthBegin(1)
        idx = pd.date_range(start=start, end=end, freq="30min", inclusive="left")

        # weights day-by-day
        daily_dhw = []
        daily_space = []
        for d in pd.date_range(start=start, end=end, freq="D", inclusive="left"):
            is_weekend = d.weekday() >= 5
            daily_dhw.append(dhw_we_48 if is_weekend else dhw_wd_48)
            daily_space.append(space_we_48 if is_weekend else space_wd_48)

        dhw_weights = np.concatenate(daily_dhw)
        space_weights = np.concatenate(daily_space)

        L = min(len(idx), len(dhw_weights))
        idx = idx[:L]
        dhw_weights = dhw_weights[:L]
        space_weights = space_weights[:L]

        dhw_useful_ts_kWh = row["dhw_useful_kWh"] * (dhw_weights / dhw_weights.sum())
        space_useful_ts_kWh = row["space_useful_kWh"] * (space_weights / space_weights.sum())
        useful_ts_kWh = dhw_useful_ts_kWh + space_useful_ts_kWh

        fuel_ts_kWh = useful_ts_kWh / boiler_eff

        out = pd.DataFrame(index=idx)
        out["heat_output_kW"] = useful_ts_kWh / DT_HOURS
        out["gas_input_kW"] = fuel_ts_kWh / DT_HOURS
        parts.append(out)

    prof = pd.concat(parts).sort_index()
    return prof


# -----------------------------
# Finance
# -----------------------------

def npv(cashflows: list[float], r: float) -> float:
    return sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))


def financial_appraisal(
    year: int,
    elec_site_kWh: float,
    heat_useful_kWh: float,
    chp_elec_gen_kWh: float,
    chp_fuel_kWh: float,
    residual_boiler_heat_kWh: float,
    elec_price_0: float,
    gas_price_0: float,
    discount_rate: float,
    elec_escalation: float,
    gas_escalation: float,
    chp_capex_eur: float,
) -> dict:
    # Baseline (75% boilers)
    baseline_gas_kWh = heat_useful_kWh / BASELINE_BOILER_EFF
    baseline_cost0 = elec_site_kWh * elec_price_0 + baseline_gas_kWh * gas_price_0

    # Option (CHP + 95% aux boilers)
    grid_import_kWh = max(elec_site_kWh - chp_elec_gen_kWh, 0.0)
    aux_gas_kWh = residual_boiler_heat_kWh / NEW_BOILER_EFF
    option_cost0 = grid_import_kWh * elec_price_0 + (chp_fuel_kWh + aux_gas_kWh) * gas_price_0

    saving0 = baseline_cost0 - option_cost0
    payback = (chp_capex_eur / saving0) if saving0 > 0 else np.inf

    cashflows = [-chp_capex_eur]
    for y in range(1, 20 + 1):
        ep = elec_price_0 * (1 + elec_escalation) ** (y - 1)
        gp = gas_price_0 * (1 + gas_escalation) ** (y - 1)
        baseline = elec_site_kWh * ep + baseline_gas_kWh * gp
        option = grid_import_kWh * ep + (chp_fuel_kWh + aux_gas_kWh) * gp
        cashflows.append(baseline - option)

    return {
        "year": year,
        "baseline_cost_year0": baseline_cost0,
        "option_cost_year0": option_cost0,
        "annual_saving_year0": saving0,
        "simple_payback_years": payback,
        "npv_20y": npv(cashflows, discount_rate),
        "grid_import_kWh": grid_import_kWh,
        "baseline_gas_kWh": baseline_gas_kWh,
        "aux_boiler_gas_kWh": aux_gas_kWh,
    }


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="CHP App (Notebook-aligned)", layout="wide")
st.title("CHP Sizing, Heat Profiles, and 20-Year Appraisal")

st.caption(
    "This app mirrors the methodology in CHP_Analysis_FIXED.ipynb: "
    "cleaned half-hourly demand, validated annual kWh (rolling pairs), "
    "weekday/weekend heat profile synthesis from monthly gas+occupancy, "
    "always-on CHP simulation, and 20-year payback/NPV."
)

with st.sidebar:
    st.header("Inputs")

    analysis_year = st.selectbox("Analysis year", [2015, 2016], index=1)
    st.info("Financial calculations are performed for the selected analysis year.")

    st.subheader("Files")
    demand_file = st.file_uploader("Upload electrical demand CSV (data_transposed.csv)", type=["csv"])
    moodle_file = st.file_uploader("Optional: upload raw Moodle export (electrical_demand_moodle.csv)", type=["csv"])
    gas_occ_file = st.file_uploader("Upload Gas + Occupancy Excel", type=["xlsx"])

    st.subheader("CHP catalog")
    chp_cat_file = st.file_uploader("Upload CHP catalog CSV (natgas_table.csv or biogas_table.csv)", type=["csv"])

    st.subheader("Prices and finance")
    elec_price_0 = st.number_input("Electricity price (€/kWh)", value=float(DEFAULT_ELEC_PRICE), step=0.01)
    gas_price_0 = st.number_input("Gas price (€/kWh)", value=float(DEFAULT_GAS_PRICE), step=0.01)
    discount_rate = st.number_input("Discount rate", value=float(DEFAULT_DISCOUNT_RATE), step=0.01, format="%.3f")
    elec_escalation = st.number_input("Electricity escalation (p.a.)", value=float(DEFAULT_ESCALATION), step=0.01, format="%.3f")
    gas_escalation = st.number_input("Gas escalation (p.a.)", value=float(DEFAULT_ESCALATION), step=0.01, format="%.3f")
    chp_capex_per_kwe = st.number_input("CHP CAPEX (€/kWe)", value=float(DEFAULT_CHP_CAPEX_PER_KWE), step=50.0)

    st.subheader("Cleaning")
    max_gap_steps = st.slider("Max interpolation gap (30-min steps)", min_value=0, max_value=48, value=12)

# -----------------------------
# Load demand series
# -----------------------------
if demand_file is None and moodle_file is None:
    st.warning("Upload an electrical demand CSV (preferred) or the raw Moodle export.")
    st.stop()

if demand_file is not None:
    df_demand = pd.read_csv(demand_file)
else:
    # Save uploaded Moodle CSV to a temp file then transpose
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
        tf.write(moodle_file.getbuffer())
        moodle_path = Path(tf.name)
    df_demand = transpose_moodle_electric_csv(moodle_path)

power_raw = load_power_timeseries(df_demand)
power = clean_power_series(power_raw, max_gap_steps=max_gap_steps)

# Diagnostics
power_year = power.loc[str(int(analysis_year))]
st.subheader("Electrical demand diagnostics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records in year (30-min)", f"{len(power_year):,}")
c2.metric("NaNs remaining", f"{int(power_year.isna().sum()):,}")
c3.metric("Mean kW", f"{float(np.nanmean(power_year)):.1f}")
c4.metric("Peak kW", f"{float(np.nanmax(power_year)):.1f}")

site_kWh_validated = validated_annual_kwh(power, year=int(analysis_year))
st.success(f"Validated annual electricity demand ({analysis_year}): {site_kWh_validated:,.2f} kWh")

# Baseload
baseload = estimate_baseload_kW(power, year=int(analysis_year))
st.write(f"Estimated electrical baseload (night {BASELOAD_TIME_START}-{BASELOAD_TIME_END}, q={BASELOAD_QUANTILE:.0%}): **{baseload:.1f} kW**")

# Plot demand (one week)
st.subheader("Electrical demand profile (example week)")
week_start = st.date_input("Week start (for demand plot)", value=pd.Timestamp(f"{analysis_year}-01-11").date())
week_end = pd.Timestamp(week_start) + pd.Timedelta(days=7)
week = power.loc[pd.Timestamp(week_start):week_end]

fig = go.Figure()
fig.add_trace(go.Scatter(x=week.index, y=week.values, mode="lines", name="Site demand (kW)"))
fig.update_layout(height=350, xaxis_title="Time", yaxis_title="kW")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Load CHP catalog
# -----------------------------
if chp_cat_file is None:
    st.warning("Upload a CHP catalog CSV (natgas_table.csv or biogas_table.csv) to proceed with CHP sizing.")
    st.stop()

cat = pd.read_csv(chp_cat_file)
req = ["Product", "Electrical Output (kWe)", "Total Heat Output (kWth)", "Fuel Input LHV (kW)"]
missing = [c for c in req if c not in cat.columns]
if missing:
    st.error(f"CHP catalog missing columns: {missing}")
    st.stop()

cat = cat.sort_values("Electrical Output (kWe)").reset_index(drop=True)

with st.sidebar:
    st.subheader("CHP selection")
    product = st.selectbox("Select CHP unit", cat["Product"].tolist(), index=min(0, len(cat)-1))

unit = cat[cat["Product"] == product].iloc[0]
chp_kwe = float(unit["Electrical Output (kWe)"])
chp_heat_kw = float(unit["Total Heat Output (kWth)"])
chp_fuel_kw = float(unit["Fuel Input LHV (kW)"])

st.subheader("Selected CHP unit")
st.write(unit.to_frame("value"))

# CHP simulation
sim = simulate_constant_chp_against_demand(power, chp_kWe=chp_kwe, year=int(analysis_year))

st.subheader("CHP electricity balance (always-on)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Site demand (kWh, validated)", f"{sim.site_demand_kWh:,.0f}")
c2.metric("CHP generation (kWh)", f"{sim.chp_gen_kWh:,.0f}")
c3.metric("Grid import (kWh)", f"{sim.grid_import_kWh:,.0f}")
c4.metric("Export (kWh)", f"{sim.export_kWh:,.0f}")

# -----------------------------
# Heat profile + CHP heat comparison
# -----------------------------
if gas_occ_file is None:
    st.warning("Upload Gas + Occupancy Excel to generate heat profiles and run the boiler/finance sections.")
    st.stop()

tmp_xlsx = Path(st.session_state.get("_gas_occ_path", ""))
# Save uploaded xlsx to a temp file for pandas
import tempfile, os
with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tf:
    tf.write(gas_occ_file.getbuffer())
    gas_occ_path = Path(tf.name)

gas_monthly, occ_monthly = tidy_monthly_gas_occ(gas_occ_path)
hp = build_heat_profile_from_monthly(gas_monthly, occ_monthly, year=int(analysis_year), boiler_eff=BASELINE_BOILER_EFF)

# Compare CHP heat vs demand
hp = hp.copy()
hp["chp_heat_kW"] = chp_heat_kw
hp["boiler_residual_kW"] = (hp["heat_output_kW"] - hp["chp_heat_kW"]).clip(lower=0)
hp["excess_heat_kW"] = (hp["chp_heat_kW"] - hp["heat_output_kW"]).clip(lower=0)

annual_heat_useful_kWh = float((hp["heat_output_kW"] * DT_HOURS).sum())
annual_residual_heat_kWh = float((hp["boiler_residual_kW"] * DT_HOURS).sum())
annual_excess_heat_kWh = float((hp["excess_heat_kW"] * DT_HOURS).sum())

st.subheader("Heat balance (useful heat)")
c1, c2, c3 = st.columns(3)
c1.metric("Annual useful heat demand (kWh)", f"{annual_heat_useful_kWh:,.0f}")
c2.metric("Annual residual boiler heat (kWh)", f"{annual_residual_heat_kWh:,.0f}")
c3.metric("Annual excess CHP heat (kWh)", f"{annual_excess_heat_kWh:,.0f}")

# Plot winter and summer example weeks
st.subheader("CHP heat vs demand (winter and summer weeks)")
winter_start = pd.Timestamp(f"{analysis_year}-01-11")
summer_start = pd.Timestamp(f"{analysis_year}-07-11")

def plot_week(start: pd.Timestamp, title: str):
    end = start + pd.Timedelta(days=7)
    w = hp.loc[start:end]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w["heat_output_kW"], mode="lines", name="Heat demand (kW)"))
    fig.add_trace(go.Scatter(x=w.index, y=w["chp_heat_kW"], mode="lines", name="CHP heat (kW)"))
    fig.add_trace(go.Scatter(x=w.index, y=w["boiler_residual_kW"], mode="lines", name="Boiler residual (kW)"))
    fig.add_trace(go.Scatter(x=w.index, y=w["excess_heat_kW"], mode="lines", name="Excess heat (kW)"))
    fig.update_layout(height=350, title=title, xaxis_title="Time", yaxis_title="kW")
    st.plotly_chart(fig, use_container_width=True)

plot_week(winter_start, "Winter week (example)")
plot_week(summer_start, "Summer week (example)")

# Daily profile plots (weekday vs weekend, and winter vs summer)
st.subheader("Daily heat profiles (weekday vs weekend; winter vs summer)")
hp_prof = hp.copy()
hp_prof["day_type"] = np.where(hp_prof.index.weekday >= 5, "Weekend", "Weekday")
hp_prof["time"] = hp_prof.index.strftime("%H:%M")
hp_prof["month"] = hp_prof.index.month

def daily_profile(months: list[int] | None, title: str):
    sub = hp_prof if months is None else hp_prof[hp_prof["month"].isin(months)]
    g = sub.groupby(["day_type", "time"])["heat_output_kW"].mean().reset_index()
    fig = go.Figure()
    for dt in ["Weekday", "Weekend"]:
        s = g[g["day_type"] == dt]
        fig.add_trace(go.Scatter(x=s["time"], y=s["heat_output_kW"], mode="lines", name=dt))
    fig.update_layout(height=350, title=title, xaxis_title="Time of day", yaxis_title="kW")
    fig.update_xaxes(tickangle=45, nticks=12)
    st.plotly_chart(fig, use_container_width=True)

daily_profile(None, "Average 24h heat profile (all months)")
daily_profile([12, 1, 2], "Winter (Dec–Feb) average 24h heat profile")
daily_profile([6, 7, 8], "Summer (Jun–Aug) average 24h heat profile")

# Boiler sizing basis
peak_residual_kW = float(hp["boiler_residual_kW"].max())
design_margin = 1.15
required_kW = peak_residual_kW * design_margin
st.subheader("Boiler sizing basis (auxiliary boilers)")
st.write(f"Peak residual boiler load: **{peak_residual_kW:.1f} kW**")
st.write(f"Design margin: **{design_margin:.2f}** → Required capacity: **{required_kW:.1f} kW**")
st.write("Typical recommendation: **2 x (required/2) kW modulating condensing boilers** for redundancy and turndown.")

# -----------------------------
# Finance (locked to selected year)
# -----------------------------
st.subheader("20-year investment appraisal (baseline vs CHP option)")
chp_capex = chp_capex_per_kwe * chp_kwe
chp_fuel_kWh = chp_fuel_kw * 8760.0  # consistent with always-on assumption and datasheet

fin = financial_appraisal(
    year=int(analysis_year),
    elec_site_kWh=site_kWh_validated,
    heat_useful_kWh=annual_heat_useful_kWh,
    chp_elec_gen_kWh=sim.chp_gen_kWh,
    chp_fuel_kWh=chp_fuel_kWh,
    residual_boiler_heat_kWh=annual_residual_heat_kWh,
    elec_price_0=float(elec_price_0),
    gas_price_0=float(gas_price_0),
    discount_rate=float(discount_rate),
    elec_escalation=float(elec_escalation),
    gas_escalation=float(gas_escalation),
    chp_capex_eur=float(chp_capex),
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline cost (Year 0)", f"€{fin['baseline_cost_year0']:,.0f}")
c2.metric("Option cost (Year 0)", f"€{fin['option_cost_year0']:,.0f}")
c3.metric("Saving (Year 0)", f"€{fin['annual_saving_year0']:,.0f}")
c4.metric("Payback", f"{fin['simple_payback_years']:.1f} y" if np.isfinite(fin['simple_payback_years']) else "∞")

st.metric("NPV (20y)", f"€{fin['npv_20y']:,.0f}")

st.caption(
    "Baseline: all electricity purchased; all heat from 75% gas boilers. "
    "Option: CHP always-on + grid import; residual heat from 95% boilers. "
    "Energy prices escalated annually; NPV discounted at the chosen rate."
)

# Sensitivity (single-axis: same escalation for gas & electricity)
st.subheader("Sensitivity: energy price escalation (same for gas & electricity)")
sens = []
for esc in [0.01, 0.02, 0.04]:
    f2 = financial_appraisal(
        year=int(analysis_year),
        elec_site_kWh=site_kWh_validated,
        heat_useful_kWh=annual_heat_useful_kWh,
        chp_elec_gen_kWh=sim.chp_gen_kWh,
        chp_fuel_kWh=chp_fuel_kWh,
        residual_boiler_heat_kWh=annual_residual_heat_kWh,
        elec_price_0=float(elec_price_0),
        gas_price_0=float(gas_price_0),
        discount_rate=float(discount_rate),
        elec_escalation=esc,
        gas_escalation=esc,
        chp_capex_eur=float(chp_capex),
    )
    sens.append({"escalation": esc, "npv_20y": f2["npv_20y"]})
sens_df = pd.DataFrame(sens)
st.dataframe(sens_df, use_container_width=True)

