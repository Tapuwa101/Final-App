#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYCSBus — Alarms & Hotspots Explorer (Streamlit)
v2.1 — Folium fallback: if Folium isn't installed, the app auto‑switches to Pydeck.
"""
import io
import math
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

# Try Folium; fall back to Pydeck if not available
FOLIUM_AVAILABLE = True
try:
    import folium  # type: ignore
    from folium.plugins import MarkerCluster, HeatMap  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    FOLIUM_AVAILABLE = False

import pydeck as pdk  # always available via requirements

# -----------------------------
# Page / theme config
# -----------------------------
st.set_page_config(
    page_title="NYC Bus Safety — Alarms & Hotspots (v2.1)",
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

TILES_DARK = 'CartoDB dark_matter'
TILES_LIGHT = 'CartoDB positron'
DECK_DARK = 'dark'
DECK_LIGHT = 'light'

BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
BOROUGH_FIX = {"NEW YORK": "Manhattan", "RICHMOND": "Staten Island"}
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
    # Handle common schemas
    if "latitude" in df.columns and "longitude" in df.columns:
        df["center_lat"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["center_lon"] = pd.to_numeric(df["longitude"], errors="coerce")
        return df
    if "centroid_lat" in df.columns and "centroid_lon" in df.columns:
        df["center_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
        df["center_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
        return df
    lat_candidates = [c for c in df.columns if c in {"lat","latitude","center_lat","centroid_lat"}]
    lon_candidates = [c for c in df.columns if c in {"lon","lng","longitude","center_lon","centroid_lon"}]
    if not lat_candidates or not lon_candidates:
        return df
    latc, lonc = lat_candidates[0], lon_candidates[0]
    df["center_lat"] = pd.to_numeric(df[latc], errors="coerce")
    df["center_lon"] = pd.to_numeric(df[lonc], errors="coerce")
    return df

def ensure_borough(df: pd.DataFrame) -> pd.DataFrame:
    for candidate in ["borough", "boro", "boro_name"]:
        if candidate in df.columns:
            b = df[candidate].astype(str).str.strip().replace(BOROUGH_FIX)
            df["borough"] = b.str.title()
            return df
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
    if "crash_date" in df.columns:
        dt = pd.to_datetime(df["crash_date"], errors="coerce", infer_datetime_format=True)
        if "crash_time" in df.columns:
            dt = pd.to_datetime(df["crash_date"] + " " + df["crash_time"], errors="coerce")
        df["__when"] = dt
        return df
    for cand in ["time_bin","time_period","date","week","period","timestamp"]:
        if cand in df.columns:
            parsed = pd.to_datetime(df[cand], errors="coerce")
            if parsed.notna().any():
                df["__when"] = parsed
                return df
    df["__when"] = pd.NaT
    return df

def week_floor(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()

def make_pydeck_hex(df_points: pd.DataFrame,
                    lat_col="center_lat",
                    lon_col="center_lon",
                    elevation_col=None,
                    radius=120,
                    center=NYC_CENTER,
                    zoom=10.5,
                    deck_style=DECK_DARK):
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom, pitch=30)
    elev = elevation_col if (elevation_col and elevation_col in df_points.columns) else None
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
        get_elevation=elev,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=deck_style,
        tooltip={"text": "{position}\nHeight ~ point density" if not elev else f"{{{elev}}}"}
    )
    return r


def make_folium_map(df_points: pd.DataFrame,
                    lat_col="center_lat",
                    lon_col="center_lon",
                    popup_cols=None,
                    heat=False,
                    zoom=10.5,
                    center=NYC_CENTER,
                    tiles=TILES_DARK):
    m = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=True)
    if heat and lat_col in df_points and lon_col in df_points:
        heat_data = df_points[[lat_col, lon_col]].dropna().values.tolist()
        if len(heat_data) > 0:
            HeatMap(heat_data, radius=12, blur=16, min_opacity=0.15).add_to(m)
    cluster = MarkerCluster().add_to(m)
    if popup_cols is None:
        popup_cols = []
    # palette based on base map
    is_light = isinstance(tiles, str) and ('positron' in tiles.lower() or 'light' in tiles.lower())
    edge = '#2b2b2b' if is_light else '#cccccc'
    fill = '#111827' if is_light else '#dddddd'
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
            color=edge,
            fill_color=fill
        ).add_to(cluster if popup_html == "" else folium.map.FeatureGroup().add_to(m))
        if popup_html:
            folium.Marker(
                location=[r[lat_col], r[lon_col]],
                icon=folium.Icon(color='darkblue' if is_light else 'lightgray', icon='info-sign'),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(cluster)
    return m

# --- Compatibility shim (accept tiles= kw on older deployments) ---
try:
    import inspect as _inspect
    _sig = _inspect.signature(make_folium_map)
    if 'tiles' not in _sig.parameters:
        _old_make_folium_map = make_folium_map
        def make_folium_map(*args, tiles=None, **kwargs):  # type: ignore[override]
            # tiles is ignored by the old implementation; function will still return a map
            return _old_make_folium_map(*args, **kwargs)
except Exception:
    pass
# ### FOLIUM MAP COMPAT SHIM ###
def make_pydeck_scatter(df_points: pd.DataFrame,
                        lat_col="center_lat",
                        lon_col="center_lon",
                        center=NYC_CENTER,
                        zoom=10.5,
                        color_from=None,
                        tooltip_cols=None,
                        deck_style=DECK_DARK):
    import pydeck as pdk
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom, pitch=30)
    def get_color(row):
        v = str(row.get(color_from, "")).lower() if color_from else ""
        # Basic palette: increase=green, decrease=red, else gray
        if "increas" in v:
            return [60, 180, 75, 180]
        if "decreas" in v:
            return [220, 50, 50, 180]
        return [160, 160, 160, 160]
    data = df_points.dropna(subset=[lat_col, lon_col]).copy()
    if color_from:
        data["__color_rgba"] = data.apply(get_color, axis=1)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=[lon_col, lat_col],
        get_fill_color="__color_rgba" if color_from else [160,160,160,160],
        get_radius=60,
        pickable=True,
    )
    tooltip_text = "\\n".join([f"{c}: {{{c}}}" for c in (tooltip_cols or [])])
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=deck_style,
                 tooltip={"text": tooltip_text} if tooltip_cols else None)
    return r

def make_folium_trend_map(df_points: pd.DataFrame,
                          lat_col="center_lat",
                          lon_col="center_lon",
                          center=NYC_CENTER,
                          zoom=10.5,
                          tooltip_cols=None, tiles=TILES_DARK):
    m = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=True)
    def color_for(v):
        v = str(v).lower()
        if "increas" in v:
            return "#2ecc71"  # green
        if "decreas" in v:
            return "#e74c3c"  # red
        return "#95a5a6"      # gray
    for _, r in df_points.dropna(subset=[lat_col, lon_col]).iterrows():
        popup_items = []
        if tooltip_cols:
            for c in tooltip_cols:
                popup_items.append(f"<b>{c}:</b> {r.get(c, '')}")
        folium.CircleMarker(
            location=[r[lat_col], r[lon_col]],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.9,
            color=color_for(r.get("trend_direction")),
            fill_color=color_for(r.get("trend_direction"))
        ).add_to(m if not popup_items else folium.Marker(
            location=[r[lat_col], r[lon_col]],
            icon=folium.Icon(color="lightgray", icon="info-sign"),
            popup=folium.Popup("<br>".join(popup_items), max_width=300)
        ).add_to(m))
    return m


# -----------------------------
# Sidebar — input mode & data
# -----------------------------
st.sidebar.title("📥 Load your data")
input_mode = st.sidebar.radio(
    "Choose input mode",
    ["Raw NYC 'Motor Vehicle Collisions - Crashes' CSV", "Precomputed CSVs (alarms/hotspots)"],
    index=0
)

engine_options = ["Pydeck (hex bins)"] if not FOLIUM_AVAILABLE else ["Folium (dark tiles)", "Pydeck (hex bins)"]
engine = st.sidebar.radio("Map engine", engine_options, index=0)
map_theme = st.sidebar.selectbox("Map theme", ["Dark", "Light"], index=1 if FOLIUM_AVAILABLE else 1)
show_heat = st.sidebar.checkbox("Add heatmap (Folium only)", value=True, disabled=not FOLIUM_AVAILABLE)

# Resolve theme -> tiles / deck style
tiles_selected = TILES_LIGHT if map_theme == "Light" else TILES_DARK
deck_selected = DECK_LIGHT if map_theme == "Light" else DECK_DARK

if not FOLIUM_AVAILABLE and "Folium" in engine:
    st.sidebar.warning("Folium is not installed in this environment. Using Pydeck instead.")

alarms_df = None
hotspots_df = None

if input_mode.startswith("Raw"):
    raw_file = st.sidebar.file_uploader("Upload the NYC collisions CSV", type=["csv"], key="raw")
    st.sidebar.caption("NYC Open Data → “Motor Vehicle Collisions - Crashes”.")
    spatial_dec = st.sidebar.slider("Spatial precision (round lat/lon)", 2, 4, 3)
    min_points_for_cell = st.sidebar.number_input("Min crashes per cell/week", 1, value=3, step=1)
    history_weeks = st.sidebar.slider("History window (weeks)", 4, 26, 8)
    z_thresh = st.sidebar.slider("Z-score threshold (alarm trigger)", 1.0, 4.0, 2.0, 0.1)

    if raw_file is not None:
        raw = pd.read_csv(raw_file, low_memory=False)
        raw = standardize_columns(raw)
        if "latitude" in raw.columns and "longitude" in raw.columns:
            raw["center_lat"] = pd.to_numeric(raw["latitude"], errors="coerce")
            raw["center_lon"] = pd.to_numeric(raw["longitude"], errors="coerce")
        raw = ensure_borough(raw)
        raw = try_parse_datetime(raw)
        raw = raw.dropna(subset=["center_lat","center_lon"])

        min_date = pd.to_datetime(raw["__when"]).min().date() if raw["__when"].notna().any() else None
        max_date = pd.to_datetime(raw["__when"]).max().date() if raw["__when"].notna().any() else None
        if min_date and max_date:
            sel_dates = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
            start_d, end_d = pd.to_datetime(sel_dates[0]), pd.to_datetime(sel_dates[1]) + pd.Timedelta(days=1)
            raw = raw[(raw["__when"] >= start_d) & (raw["__when"] < end_d)]

        raw["lat_r"] = raw["center_lat"].round(spatial_dec)
        raw["lon_r"] = raw["center_lon"].round(spatial_dec)
        raw["cell_id"] = raw["lat_r"].astype(str) + "," + raw["lon_r"].astype(str)
        raw["week_start"] = week_floor(raw["__when"])

        agg = (raw.groupby(["cell_id","lat_r","lon_r","borough","week_start"], dropna=False)
               .size().reset_index(name="crash_count")).sort_values(["cell_id","week_start"])

        def label_cell(group):
            group = group.sort_values("week_start").copy()
            group["mean_prev"] = group["crash_count"].rolling(window=history_weeks, min_periods=3).mean().shift(1)
            group["std_prev"] = group["crash_count"].rolling(window=history_weeks, min_periods=3).std(ddof=0).shift(1)
            group["zscore"] = (group["crash_count"] - group["mean_prev"]) / group["std_prev"]
            group.loc[group["std_prev"].isna() | (group["std_prev"] == 0), "zscore"] = 0.0
            group["inc_3w"] = group["crash_count"].diff().rolling(3).sum() > 0
            cond = (group["crash_count"] >= min_points_for_cell) & (group["zscore"] >= z_thresh)
            alarm = np.where(cond & (~group["inc_3w"].shift(1).fillna(False)), "New",
                     np.where(cond & group["inc_3w"], "Intensifying",
                     np.where(cond & (group["crash_count"].rolling(4).min() >= group["mean_prev"]), "Persistent", None)))
            group["alarm_type"] = alarm
            return group

        agg = agg.groupby("cell_id", group_keys=False).apply(label_cell)
        alarms_df = agg[agg["alarm_type"].notna()].copy()
        alarms_df["center_lat"] = alarms_df["lat_r"]
        alarms_df["center_lon"] = alarms_df["lon_r"]
        alarms_df["time_bin"] = alarms_df["week_start"].dt.date.astype(str)
        alarms_df["severity"] = alarms_df["zscore"].round(2)
        alarms_df["h3_cell"] = alarms_df["cell_id"]
        alarms_df = alarms_df[["center_lat","center_lon","borough","time_bin","alarm_type","severity","h3_cell"]]

        totals = (agg.groupby(["cell_id","lat_r","lon_r","borough"], dropna=False)["crash_count"]
                  .sum().reset_index())
        q85 = totals["crash_count"].quantile(0.85) if len(totals) > 0 else 0
        totals["hotspot_type"] = np.where(totals["crash_count"] >= q85, "Hot Spot", "Normal")
        totals["center_lat"] = totals["lat_r"]
        totals["center_lon"] = totals["lon_r"]
        hotspots_df = totals.rename(columns={"crash_count":"score"})
        hotspots_df = hotspots_df[["center_lat","center_lon","borough","hotspot_type","score"]]

else:
    alarms_file = st.sidebar.file_uploader("Alarms CSV (required)", type=["csv"], key="alarms")
    hotspots_file = st.sidebar.file_uploader("Hotspots CSV (optional)", type=["csv"], key="hotspots")
    trends_file = st.sidebar.file_uploader("Trends CSV (optional)", type=["csv"], key="trends")
    if alarms_file is not None:
        alarms_df = pd.read_csv(alarms_file)
        alarms_df = standardize_columns(alarms_df)
        alarms_df = coerce_latlon(alarms_df)
        alarms_df = ensure_borough(alarms_df)
    # Normalize alarm_type labels (e.g., 'Intensifying Hotspot' -> 'Intensifying')
    if 'alarm_type' in alarms_df.columns:
        at = alarms_df['alarm_type'].astype(str).str.lower()
        mapped = []
        for v in at:
            if 'new' in v:
                mapped.append('New')
            elif 'intens' in v:
                mapped.append('Intensifying')
            elif 'persist' in v:
                mapped.append('Persistent')
            else:
                mapped.append(v.title())
        alarms_df['alarm_type'] = mapped
    if hotspots_file is not None:
        hotspots_df = pd.read_csv(hotspots_file)
        hotspots_df = standardize_columns(hotspots_df)
        hotspots_df = coerce_latlon(hotspots_df)
        hotspots_df = ensure_borough(hotspots_df)
        hotspots_df = try_parse_datetime(hotspots_df)

# -----------------------------
# Stop early if no data
# -----------------------------
if alarms_df is None or len(alarms_df) == 0:
    st.warning("Upload data to begin. Choose **Raw NYC CSV** or **Precomputed** in the sidebar.")
    if not FOLIUM_AVAILABLE:
        st.info("Folium is not installed here. The app will use **Pydeck** automatically.")
    st.stop()

# -----------------------------
# Filters
# -----------------------------
alarm_types = sorted([t for t in alarms_df.get("alarm_type", pd.Series(dtype=str)).dropna().astype(str).str.title().unique()]) or ["New","Intensifying","Persistent"]
boro_present = BOROUGH_ORDER

when_series = pd.to_datetime(alarms_df.get("__when", alarms_df.get("time_bin", pd.Series(dtype=str))), errors="coerce")
min_date = pd.to_datetime(when_series.min()).date() if when_series.notna().any() else None
max_date = pd.to_datetime(when_series.max()).date() if when_series.notna().any() else None

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")
sel_boroughs = st.sidebar.multiselect("Boroughs", options=BOROUGH_ORDER, default=BOROUGH_ORDER)
sel_alarm_types = st.sidebar.multiselect("Alarm types", options=alarm_types, default=alarm_types)
sel_dates = None
if min_date and max_date:
    sel_dates = st.sidebar.date_input("Date range (for display)", value=(min_date, max_date), min_value=min_date, max_value=max_date)

df_f = alarms_df.copy()
if "borough" in df_f.columns:
    df_f = df_f[df_f["borough"].isin(sel_boroughs)]
df_f = df_f[df_f["alarm_type"].str.title().isin(sel_alarm_types)] if "alarm_type" in df_f.columns else df_f
if sel_dates is not None:
    time_col = "__when" if "__when" in df_f.columns else "time_bin"
    if time_col in df_f.columns:
        t = pd.to_datetime(df_f[time_col], errors="coerce")
        start_d, end_d = pd.to_datetime(sel_dates[0]), pd.to_datetime(sel_dates[1]) + pd.Timedelta(days=1)
        df_f = df_f[(t >= start_d) & (t < end_d)]

# -----------------------------
# Header + KPIs
# -----------------------------
st.title("🗽 NYC Bus Safety — Alarms & Hotspots (v2.1)")
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
tabs = st.tabs(["🌆 Citywide map", "🗺️ Borough maps", "📈 Trends", "📄 Data"])

with tabs[0]:
    st.subheader("Citywide alarms")
    if FOLIUM_AVAILABLE and engine.startswith("Folium"):
        popup_cols = ["borough", "alarm_type", "time_bin", "h3_cell", "severity"]
        m = make_folium_map(df_f, popup_cols=popup_cols, heat=show_heat, center=NYC_CENTER, zoom=10.6, tiles=tiles_selected)
        st_folium(m, width=None, height=560)
    else:
        st.pydeck_chart(make_pydeck_hex(df_f, elevation_col=None, center=NYC_CENTER, zoom=10.6, deck_style=deck_selected))

    if 'hotspots_df' in globals() and hotspots_df is not None and len(hotspots_df) > 0:
        st.markdown("—")
        st.subheader("Citywide hotspots")
        hs = hotspots_df.copy()
        if "borough" in hs.columns:
            hs = hs[hs["borough"].isin(sel_boroughs)]
        hs = hs.dropna(subset=["center_lat","center_lon"]).head(10000)
        if FOLIUM_AVAILABLE and engine.startswith("Folium"):
            popup_cols = ["borough","hotspot_type","score"]
            mh = make_folium_map(hs, popup_cols=popup_cols, heat=False, center=NYC_CENTER, zoom=10.6, tiles=tiles_selected)
            st_folium(mh, width=None, height=560)
        else:
            st.pydeck_chart(make_pydeck_hex(hs, elevation_col="score", center=NYC_CENTER, zoom=10.6, deck_style=deck_selected))

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
                center = NYC_CENTER if df_b.empty else (df_b["center_lat"].median(), df_b["center_lon"].median())
                if FOLIUM_AVAILABLE and engine.startswith("Folium"):
                    popup_cols = ["hotspot_type","score"]
                    mb = make_folium_map(df_b, popup_cols=popup_cols, heat=False, center=center, zoom=11.2, tiles=tiles_selected)
                    st_folium(mb, width=None, height=520)
                else:
                    st.pydeck_chart(make_pydeck_hex(df_b, elevation_col="score", center=center, zoom=11.2, deck_style=deck_selected))

with tabs[2]:
    st.subheader("Filtered alarms (data)")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    csv = df_f.to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ Download filtered CSV", csv, file_name="filtered_alarms.csv", mime="text/csv")
    c2.download_button("⬇️ Download filtered JSON", df_f.to_json(orient="records").encode("utf-8"),
                       file_name="filtered_alarms.json", mime="application/json")

st.markdown("---")
with st.expander("🧰 Troubleshooting & tips"):
    st.write("""
- **Folium missing?** This deployment auto‑switches to **Pydeck**. To enable Folium, make sure `folium` and `streamlit-folium` are in `requirements.txt`, then reboot.
- Use **Raw NYC CSV** mode to derive weekly alarms and hotspots from the collisions dataset. No H3 required.
- **Dark theme** throughout. Switch to **Pydeck (hex bins)** for aggregated 3D if points feel dense.
""")
