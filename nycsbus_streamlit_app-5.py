#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYCSBus — Alarms & Hotspots Explorer (Streamlit)
v2 — adds support for raw NYC "Motor Vehicle Collisions - Crashes" CSV

This version still supports precomputed alarms/hotspots CSVs,
but can ALSO ingest the raw NYC collisions dataset and derive
simple "alarms" + "hotspots" on the fly (no H3 dependency).
"""
import io
import math
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

# Folium for tile maps; streamlit_folium to embed
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

# Optional: pydeck hex layer (does not need Mapbox token for basic usage)
import pydeck as pdk

# -----------------------------
# Page / theme config
# -----------------------------
st.set_page_config(
    page_title="NYC Bus Safety — Alarms & Hotspots (v2)",
    layout="wide",
    page_icon="🗽"
)

# Minimal dark UI polish
st.markdown(
    """
    <style>
      h1, h2, h3 { font-weight: 700; letter-spacing: .2px; }
      .stMetric { background: rgba(255,255,255,0.04) !important; border-radius: 16px; padding: 12px; }
      .dataframe tbody tr:hover { background: rgba(255,255,255,0.04); }
      section[data-testid="stSidebar"] { width: 360px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------
NYC_CENTER = (40.7128, -74.0060)

BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
BOROUGH_FIX = {
    "NEW YORK": "Manhattan",
    "RICHMOND": "Staten Island",
}
BOROUGH_BBOX = {
    "Manhattan":     {"lat": (40.699, 40.882), "lon": (-74.019, -73.906)},
    "Brooklyn":      {"lat": (40.560, 40.739), "lon": (-74.040, -73.856)},
    "Queens":        {"lat": (40.540, 40.800), "lon": (-73.960, -73.700)},
    "Bronx":         {"lat": (40.790, 40.920), "lon": (-73.933, -73.765)},
    "Staten Island": {"lat": (40.480, 40.650), "lon": (-74.250, -74.050)},
}

@st.cache_data(show_spinner=False)
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols).copy()
    return df

def coerce_latlon(df: pd.DataFrame) -> pd.DataFrame:
    lat_candidates = [c for c in df.columns if c in {"lat","latitude","center_lat","latitude_(y)"}]
    lon_candidates = [c for c in df.columns if c in {"lon","lng","longitude","center_lon","longitude_(x)"}]
    # Special-case the NYC raw dataset
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_candidates = ["latitude"]
        lon_candidates = ["longitude"]
    if not lat_candidates or not lon_candidates:
        return df
    latc, lonc = lat_candidates[0], lon_candidates[0]
    df["center_lat"] = pd.to_numeric(df[latc], errors="coerce")
    df["center_lon"] = pd.to_numeric(df[lonc], errors="coerce")
    return df

def ensure_borough(df: pd.DataFrame) -> pd.DataFrame:
    for candidate in ["borough", "boro", "boro_name"]:
        if candidate in df.columns:
            b = df[candidate].astype(str).str.strip()
            # NYC raw is UPPERCASE (BROOKLYN, QUEENS, etc.); fix aliases
            b = b.replace(BOROUGH_FIX)
            df["borough"] = b.str.title()
            return df
    # bbox guess if missing
    if {"center_lat","center_lon"}.issubset(df.columns):
        def guess_boro(row):
            lat, lon = row["center_lat"], row["center_lon"]
            if pd.isna(lat) or pd.isna(lon):
                return None
            for bb, box in BOROUGH_BBOX.items():
                if (box["lat"][0] <= lat <= box["lat"][1]) and (box["lon"][0] <= lon <= box["lon"][1]):
                    return bb
            return None
        df["borough"] = df.apply(guess_boro, axis=1)
    return df

def try_parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # NYC raw has CRASH DATE + CRASH TIME (lowercase after standardize)
    if "crash_date" in df.columns:
        try:
            dt = pd.to_datetime(df["crash_date"], errors="coerce", infer_datetime_format=True)
            if "crash_time" in df.columns:
                # combine date + time when possible
                tm = pd.to_datetime(df["crash_time"], format="%H:%M", errors="coerce").dt.time
                # If time parsing fails, keep date only
                dt = pd.to_datetime(df["crash_date"] + " " + df["crash_time"], errors="coerce")
            df["__when"] = dt
            return df
        except Exception:
            pass
    # Fallbacks
    for cand in ["time_bin","date","week","period","timestamp"]:
        if cand in df.columns:
            parsed = pd.to_datetime(df[cand], errors="coerce")
            if parsed.notna().any():
                df["__when"] = parsed
                return df
    df["__when"] = pd.NaT
    return df

def week_floor(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    # use Monday as week start
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()

def kpi_card(label: str, value, help_text: str=""):
    st.metric(label, value, help=help_text)

def make_folium_map(df_points: pd.DataFrame,
                    lat_col="center_lat",
                    lon_col="center_lon",
                    popup_cols=None,
                    heat=False,
                    zoom=10.5,
                    center=NYC_CENTER):
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter", control_scale=True)
    if heat and lat_col in df_points and lon_col in df_points:
        heat_data = df_points[[lat_col, lon_col]].dropna().values.tolist()
        if len(heat_data) > 0:
            HeatMap(heat_data, radius=12, blur=16, min_opacity=0.15).add_to(m)

    cluster = MarkerCluster().add_to(m)
    if popup_cols is None:
        popup_cols = []

    for _, r in df_points.dropna(subset=[lat_col, lon_col]).iterrows():
        popup_items = []
        for c in popup_cols:
            if c in df_points.columns:
                popup_items.append(f"<b>{c}:</b> {r.get(c, '')}")
        popup_html = "<br>".join(popup_items) if popup_items else ""
        folium.CircleMarker(
            location=[r[lat_col], r[lon_col]],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.85,
            color="#cccccc",
            fill_color="#dddddd"
        ).add_to(cluster if popup_html == "" else folium.map.FeatureGroup().add_to(m))
        if popup_html:
            folium.Marker(
                location=[r[lat_col], r[lon_col]],
                icon=folium.Icon(color="lightgray", icon="info-sign"),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(cluster)
    return m

def make_pydeck_hex(df_points: pd.DataFrame,
                    lat_col="center_lat",
                    lon_col="center_lon",
                    elevation_col=None,
                    radius=120,
                    center=NYC_CENTER,
                    zoom=10.5):
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom, pitch=30)
    elev = elevation_col if elevation_col in df_points.columns else None
    layer = pdk.Layer(
        "HexagonLayer",
        data=df_points.dropna(subset=[lat_col, lon_col]),
        get_position=[lon_col, lat_col],
        auto_highlight=True,
        elevation_scale=20,
        pickable=True,
        elevation_range=[0, 1000],
        extruded=True,
        coverage=1,
        radius=radius,
        get_elevation=elevation_col if elev else None,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "{position}\nHeight ~ point density" if not elev else f"{{{elev}}}"}
    )
    return r

# -----------------------------
# Sidebar — input mode & data
# -----------------------------
st.sidebar.title("📥 Load your data")
input_mode = st.sidebar.radio(
    "Choose input mode",
    ["Precomputed CSVs (alarms/hotspots)", "Raw NYC 'Motor Vehicle Collisions - Crashes' CSV"],
    index=1
)

engine = st.sidebar.radio("Map engine", ["Folium (dark tiles)", "Pydeck (hex bins)"], index=0)
show_heat = st.sidebar.checkbox("Add heatmap (Folium only)", value=True)

alarms_df = None
hotspots_df = None

if input_mode.startswith("Precomputed"):
    alarms_file = st.sidebar.file_uploader("Alarms CSV (required)", type=["csv"], key="alarms")
    hotspots_file = st.sidebar.file_uploader("Hotspots CSV (optional)", type=["csv"], key="hotspots")
    trends_file = st.sidebar.file_uploader("Trends CSV (optional)", type=["csv"], key="trends")

    with st.sidebar.expander("ℹ️ Expected columns & tips", expanded=False):
        st.write("""
- **Alarms CSV**: `center_lat`, `center_lon`, `borough` (or `boro`), `alarm_type`, and `time_bin`/`date`.
  Optional: `h3_cell`, `severity` or `score`.
- **Hotspots CSV**: `center_lat`, `center_lon`, `borough`, `hotspot_type`, `gi_star`/`score`.
""")

    if alarms_file is not None:
        alarms_df = pd.read_csv(alarms_file)
        alarms_df = standardize_columns(alarms_df)
        alarms_df = coerce_latlon(alarms_df)
        alarms_df = ensure_borough(alarms_df)
        alarms_df = try_parse_datetime(alarms_df)

    if hotspots_file is not None:
        hotspots_df = pd.read_csv(hotspots_file)
        hotspots_df = standardize_columns(hotspots_df)
        hotspots_df = coerce_latlon(hotspots_df)
        hotspots_df = ensure_borough(hotspots_df)
        hotspots_df = try_parse_datetime(hotspots_df)

else:
    raw_file = st.sidebar.file_uploader("Upload the NYC collisions CSV", type=["csv"], key="raw")
    st.sidebar.caption("Tip: On NYC Open Data this is called “Motor Vehicle Collisions - Crashes”.")
    # Controls for deriving alarms/hotspots
    spatial_dec = st.sidebar.slider("Spatial precision (round lat/lon)", min_value=2, max_value=4, value=3, help="3 ≈ 110 m; 2 ≈ 1.1 km; 4 ≈ 11 m")
    min_points_for_cell = st.sidebar.number_input("Min crashes per cell/week to consider", min_value=1, value=3, step=1)
    history_weeks = st.sidebar.slider("History window (weeks) for baseline", min_value=4, max_value=26, value=8)
    z_thresh = st.sidebar.slider("Z-score threshold (alarm trigger)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

    if raw_file is not None:
        raw = pd.read_csv(raw_file, low_memory=False)
        raw = standardize_columns(raw)
        # Map columns from NYC schema
        if "latitude" in raw.columns and "longitude" in raw.columns:
            raw["center_lat"] = pd.to_numeric(raw["latitude"], errors="coerce")
            raw["center_lon"] = pd.to_numeric(raw["longitude"], errors="coerce")
        raw = ensure_borough(raw)
        raw = try_parse_datetime(raw)
        raw = raw.dropna(subset=["center_lat","center_lon"])

        # Date filter (do early to speed up)
        min_date = pd.to_datetime(raw["__when"]).min().date() if raw["__when"].notna().any() else None
        max_date = pd.to_datetime(raw["__when"]).max().date() if raw["__when"].notna().any() else None
        if min_date and max_date:
            sel_dates = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            start_d, end_d = pd.to_datetime(sel_dates[0]), pd.to_datetime(sel_dates[1]) + pd.Timedelta(days=1)
            raw = raw[(raw["__when"] >= start_d) & (raw["__when"] < end_d)]

        # Derive a lightweight cell id by rounding lat/lon
        raw["lat_r"] = raw["center_lat"].round(spatial_dec)
        raw["lon_r"] = raw["center_lon"].round(spatial_dec)
        raw["cell_id"] = raw["lat_r"].astype(str) + "," + raw["lon_r"].astype(str)
        raw["week_start"] = week_floor(raw["__when"])

        # Aggregate to cell-week counts
        agg = (raw
               .groupby(["cell_id","lat_r","lon_r","borough","week_start"], dropna=False)
               .size()
               .reset_index(name="crash_count"))

        # Compute baseline stats and alarms per cell
        agg = agg.sort_values(["cell_id","week_start"]).reset_index(drop=True)

        # rolling baseline: previous `history_weeks` only (exclude current week)
        def label_cell(group):
            group = group.sort_values("week_start").copy()
            # rolling mean/std over prior weeks
            group["mean_prev"] = group["crash_count"].rolling(window=history_weeks, min_periods=3).mean().shift(1)
            group["std_prev"] = group["crash_count"].rolling(window=history_weeks, min_periods=3).std(ddof=0).shift(1)
            group["zscore"] = (group["crash_count"] - group["mean_prev"]) / group["std_prev"]
            group.loc[group["std_prev"].isna() | (group["std_prev"] == 0), "zscore"] = 0.0

            # simple trend flags
            group["inc_3w"] = group["crash_count"].diff().rolling(3).sum() > 0

            # classify
            cond = (group["crash_count"] >= min_points_for_cell) & (group["zscore"] >= z_thresh)
            alarm = np.where(cond & (~group["inc_3w"].shift(1).fillna(False)), "New",
                     np.where(cond & group["inc_3w"], "Intensifying",
                     np.where(cond & (group["crash_count"].rolling(4).min() >= group["mean_prev"]), "Persistent", None)))
            group["alarm_type"] = alarm
            return group

        agg = agg.groupby("cell_id", group_keys=False).apply(label_cell)

        # Build "alarms_df" with 1 row per cell-week where an alarm fired
        alarms_df = agg[agg["alarm_type"].notna()].copy()
        alarms_df["center_lat"] = alarms_df["lat_r"]
        alarms_df["center_lon"] = alarms_df["lon_r"]
        alarms_df["time_bin"] = alarms_df["week_start"].dt.date.astype(str)
        alarms_df["severity"] = alarms_df["zscore"].round(2)
        alarms_df["h3_cell"] = alarms_df["cell_id"]  # placeholder for tooltip
        alarms_df = alarms_df[["center_lat","center_lon","borough","time_bin","alarm_type","severity","h3_cell"]]

        # Build a simple hotspots_df: sum counts over the selected date range; top 15% as "Hot Spot"
        totals = (agg.groupby(["cell_id","lat_r","lon_r","borough"], dropna=False)["crash_count"]
                  .sum().reset_index())
        if len(totals) > 0:
            q85 = totals["crash_count"].quantile(0.85)
        else:
            q85 = 0
        totals["hotspot_type"] = np.where(totals["crash_count"] >= q85, "Hot Spot", "Normal")
        totals["center_lat"] = totals["lat_r"]
        totals["center_lon"] = totals["lon_r"]
        hotspots_df = totals.rename(columns={"crash_count":"score"})[["center_lat","center_lon","borough","hotspot_type","score"]]

# -----------------------------
# Filters
# -----------------------------
if alarms_df is None:
    st.warning("Upload data to begin. Choose **Precomputed** or **Raw NYC CSV** in the sidebar.")
    st.stop()

# Alarm types
if "alarm_type" in alarms_df.columns:
    alarm_types = sorted([t for t in alarms_df["alarm_type"].dropna().astype(str).str.title().unique()])
else:
    alarm_types = []
    alarms_df["alarm_type"] = np.nan

# Boroughs
from collections import OrderedDict
if "borough" in alarms_df.columns:
    boro_present = [b for b in BOROUGH_ORDER if (alarms_df["borough"] == b).any()]
    if not boro_present:
        boro_present = sorted([str(x) for x in alarms_df["borough"].dropna().unique()])
else:
    boro_present = BOROUGH_ORDER

# Time range
when_series = pd.to_datetime(alarms_df.get("__when", alarms_df.get("time_bin", pd.Series(dtype=str))), errors="coerce")
min_date = pd.to_datetime(when_series.min()).date() if when_series.notna().any() else None
max_date = pd.to_datetime(when_series.max()).date() if when_series.notna().any() else None

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")
sel_boroughs = st.sidebar.multiselect("Boroughs", options=boro_present or BOROUGH_ORDER, default=boro_present or BOROUGH_ORDER)
sel_alarm_types = st.sidebar.multiselect("Alarm types", options=alarm_types or ["New","Intensifying","Persistent"], default=alarm_types or ["New","Intensifying","Persistent"])
if min_date and max_date:
    sel_dates = st.sidebar.date_input("Date range (for display)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
else:
    sel_dates = None

# Apply filters
df_f = alarms_df.copy()
if sel_boroughs and "borough" in df_f.columns:
    df_f = df_f[df_f["borough"].isin(sel_boroughs)]
if sel_alarm_types and "alarm_type" in df_f.columns:
    df_f = df_f[df_f["alarm_type"].str.title().isin(sel_alarm_types)]
if sel_dates is not None:
    # apply only if we can parse
    time_col = "__when" if "__when" in df_f.columns else "time_bin"
    if time_col in df_f.columns:
        t = pd.to_datetime(df_f[time_col], errors="coerce")
        start_d, end_d = pd.to_datetime(sel_dates[0]), pd.to_datetime(sel_dates[1]) + pd.Timedelta(days=1)
        df_f = df_f[(t >= start_d) & (t < end_d)]

# -----------------------------
# Header + KPIs
# -----------------------------
st.title("🗽 NYC Bus Safety — Alarms & Hotspots (v2)")
st.caption("Dark‑themed, interactive view of alarms citywide and hotspot clusters per borough.")


k1, k2, k3, k4 = st.columns(4)
k1.metric("Total alarms (filtered)", f"{len(df_f):,}")
k2.metric("Boroughs in view", f"{df_f['borough'].nunique() if 'borough' in df_f else 0}")
k3.metric("Alarm types in view", f"{df_f['alarm_type'].nunique() if 'alarm_type' in df_f else 0}")
if "time_bin" in df_f.columns and df_f["time_bin"].notna().any():
    rng = f"{pd.to_datetime(df_f['time_bin']).min().date()} → {pd.to_datetime(df_f['time_bin']).max().date()}"
else:
    rng = "n/a"
k4.metric("Date span", rng)

with st.expander("📊 Quick distributions"):
    c1, c2 = st.columns(2)
    if "alarm_type" in df_f.columns and df_f["alarm_type"].notna().any():
        a_counts = df_f["alarm_type"].str.title().value_counts().rename_axis("Alarm Type").reset_index(name="Count")
        c1.bar_chart(a_counts.set_index("Alarm Type"))
    if "borough" in df_f.columns and df_f["borough"].notna().any():
        b_counts = df_f["borough"].value_counts().rename_axis("Borough").reset_index(name="Count")
        c2.bar_chart(b_counts.set_index("Borough"))

# -----------------------------
# Maps
# -----------------------------
tabs = st.tabs(["🌆 Citywide map", "🗺️ Borough maps", "📄 Data"])

# Citywide map
with tabs[0]:
    st.subheader("Citywide alarms")
    if engine.startswith("Folium"):
        popup_cols = ["borough", "alarm_type", "time_bin", "h3_cell", "severity"]
        m = make_folium_map(df_f, popup_cols=popup_cols, heat=show_heat, center=NYC_CENTER, zoom=10.6)
        st_folium(m, width=None, height=560)
    else:
        st.pydeck_chart(make_pydeck_hex(df_f, elevation_col=None, center=NYC_CENTER, zoom=10.6))

    if 'hotspots_df' in globals() and hotspots_df is not None and len(hotspots_df) > 0:
        st.markdown("—")
        st.subheader("Citywide hotspots (derived)")
        hs = hotspots_df.copy()
        if sel_boroughs and "borough" in hs.columns:
            hs = hs[hs["borough"].isin(sel_boroughs)]
        hs = hs.dropna(subset=["center_lat","center_lon"]).head(10000)
        if engine.startswith("Folium"):
            popup_cols = ["borough","hotspot_type","score"]
            mh = make_folium_map(hs, popup_cols=popup_cols, heat=False, center=NYC_CENTER, zoom=10.6)
            st_folium(mh, width=None, height=560)
        else:
            st.pydeck_chart(make_pydeck_hex(hs, elevation_col="score", center=NYC_CENTER, zoom=10.6))

# Per-borough maps
with tabs[1]:
    st.subheader("Per‑borough hotspot views")
    if 'hotspots_df' not in globals() or hotspots_df is None or len(hotspots_df) == 0:
        st.info("Hotspots require either a **Hotspots CSV** or a **Raw NYC CSV** to derive them.")
    else:
        sub_tabs = st.tabs(BOROUGH_ORDER)
        for i, b in enumerate(BOROUGH_ORDER):
            with sub_tabs[i]:
                df_b = hotspots_df[hotspots_df["borough"] == b].dropna(subset=["center_lat","center_lon"])
                st.write(f"Hotspots in **{b}** — {len(df_b):,} points")
                center = NYC_CENTER
                if not df_b.empty:
                    center = (df_b["center_lat"].median(), df_b["center_lon"].median())
                if engine.startswith("Folium"):
                    popup_cols = ["hotspot_type","score"]
                    mb = make_folium_map(df_b, popup_cols=popup_cols, heat=False, center=center, zoom=11.2)
                    st_folium(mb, width=None, height=520)
                else:
                    st.pydeck_chart(make_pydeck_hex(df_b, elevation_col="score", center=center, zoom=11.2))

# Data table + downloads
with tabs[2]:
    st.subheader("Filtered alarms (data)")
    st.caption("Tip: Use the column header menu to filter/search within the table.")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    csv = df_f.to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ Download filtered CSV", csv, file_name="filtered_alarms.csv", mime="text/csv")
    c2.download_button("⬇️ Download filtered JSON", df_f.to_json(orient="records").encode("utf-8"),
                       file_name="filtered_alarms.json", mime="application/json")

st.markdown("---")
with st.expander("🧰 Troubleshooting & tips"):
    st.write("""
- Use **Raw NYC CSV** mode to drop in the collisions dataset directly; the app derives per‑week alarms and hotspots without requiring H3.
- **Spatial precision** controls the grid size used to cluster crashes into “cells” (3 ≈ 110 m is a good start).
- **Alarms** are a lightweight heuristic: cell-week count vs prior-week baseline using a rolling mean/std and a z‑score threshold.
- **Pydeck (hex bins)** gives a clean, aggregated 3D view if point clouds feel too dense.
- **Dark theme:** Folium uses **CartoDB Dark Matter** tiles so points pop against a dark map.
""")
