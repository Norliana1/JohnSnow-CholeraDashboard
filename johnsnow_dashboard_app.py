import io
import os
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from PIL import Image

import folium
from folium.plugins import HeatMap, MarkerCluster

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import streamlit as st
from streamlit_folium import st_folium

# ---------------- PATHS ----------------
DATA_DIR = Path("data")
OUT_DIR  = Path("Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_HTML_PATH = OUT_DIR / "final_cholera_map.html"

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_csv_safe(fp: Path):
    if not fp.exists():
        raise FileNotFoundError(f"Required file not found: {fp}")
    return pd.read_csv(fp)

@st.cache_data(show_spinner=False)
def load_data():
    deaths = load_csv_safe(DATA_DIR / "deaths_by_bldg.csv")
    pumps  = load_csv_safe(DATA_DIR / "pumps.csv")
    sewer  = load_csv_safe(DATA_DIR / "sewergrates_ventilators.csv")
    return deaths, pumps, sewer

@st.cache_data(show_spinner=False)
def prepare_gdfs(deaths, pumps, sewer, xcol="COORD_X", ycol="COORD_Y", crs_proj="EPSG:27700"):
    # ensure columns exist
    for df, name in [(deaths, "deaths"), (pumps, "pumps"), (sewer, "sewer")]:
        if xcol not in df.columns or ycol not in df.columns:
            raise KeyError(f"Expected columns '{xcol}' and '{ycol}' in {name} csv.")

    # numeric
    deaths = deaths.copy()
    pumps  = pumps.copy()
    sewer  = sewer.copy()

    deaths[xcol] = pd.to_numeric(deaths[xcol], errors="coerce")
    deaths[ycol] = pd.to_numeric(deaths[ycol], errors="coerce")
    pumps[xcol]  = pd.to_numeric(pumps[xcol], errors="coerce")
    pumps[ycol]  = pd.to_numeric(pumps[ycol], errors="coerce")
    sewer[xcol]  = pd.to_numeric(sewer[xcol], errors="coerce")
    sewer[ycol]  = pd.to_numeric(sewer[ycol], errors="coerce")

    # drop missing
    deaths = deaths.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    pumps  = pumps.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    sewer  = sewer.dropna(subset=[xcol, ycol]).reset_index(drop=True)

    deaths_gdf = gpd.GeoDataFrame(
        deaths, geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]), crs=crs_proj
    )
    pumps_gdf = gpd.GeoDataFrame(
        pumps, geometry=gpd.points_from_xy(pumps[xcol], pumps[ycol]), crs=crs_proj
    )
    sewer_gdf = gpd.GeoDataFrame(
        sewer, geometry=gpd.points_from_xy(sewer[xcol], sewer[ycol]), crs=crs_proj
    )

    return deaths_gdf, pumps_gdf, sewer_gdf
# =========================================================
# BUILD FOLIUM MAP
# =========================================================
def build_folium_map(
    deaths_gdf,
    pumps_gdf,
    sewer_gdf,
    show_deaths=True,
    show_pumps=True,
    show_sewer=True,
    show_heat=True,
    bandwidth=50,
):

    # ------------------------------------------------------
    # Reproject to WGS84 for Folium
    # ------------------------------------------------------
    deaths_wgs = deaths_gdf.to_crs(epsg=4326)
    pumps_wgs  = pumps_gdf.to_crs(epsg=4326)
    sewer_wgs  = sewer_gdf.to_crs(epsg=4326)

    # ------------------------------------------------------
    # Create map center
    # ------------------------------------------------------
    center = [
        deaths_wgs.geometry.y.mean(),
        deaths_wgs.geometry.x.mean()
    ]

    m = folium.Map(
        location=center,
        zoom_start=16,
        tiles="CartoDB Positron"
    )

    # ------------------------------------------------------
    # Add Cholera Deaths
    # ------------------------------------------------------
    if show_deaths:
        deaths_fg = folium.FeatureGroup(
            name="Cholera Deaths", show=True
        )

        for idx, r in deaths_wgs.iterrows():
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=2,
                color="#444444",
                fill=True,
                fill_color="#222222",
                fill_opacity=0.8,
                popup=f"Death point #{idx+1}"
            ).add_to(deaths_fg)

        deaths_fg.add_to(m)

    # ------------------------------------------------------
    # Add Heatmap
    # ------------------------------------------------------
    if show_heat:
        coords = [
            [p.geometry.y, p.geometry.x] 
            for _, p in deaths_wgs.iterrows()
        ]
        HeatMap(coords, radius=25, blur=20).add_to(
            folium.FeatureGroup(
                name="Cholera Heatmap", show=True
            ).add_to(m)
        )

    # ------------------------------------------------------
    # PUMPS
    # ------------------------------------------------------
    if show_pumps:

        pump_group = folium.FeatureGroup(
            name="Water Pumps", show=True
        )

        # Bins & labels
        bins = [0, 10, 25, 50, 100, 200, np.inf]
        labels = ["0-10", "10-25", "25-50", "50-100", "100-200", ">200"]

        MAX_BAR_HEIGHT = 70  # px

        # reset indexing
        pumps_reset     = pumps_gdf.reset_index(drop=True)
        pumps_wgs_reset = pumps_wgs.reset_index(drop=True)

        for pid, prow in pumps_reset.iterrows():

            # compute distances to deaths
            dists = deaths_gdf.geometry.distance(prow.geometry)

            # histogram
            binned = pd.cut(dists, bins=bins, labels=labels)
            summary = (
                binned.value_counts()
                .reindex(labels)
                .fillna(0)
                .astype(int)
            )

            max_val = summary.values.max() if summary.values.max() > 0 else 1

            # build inline bar chart
            bars_html = ""
            for val, lbl in zip(summary.values, labels):
                bar_height = int((val / max_val) * MAX_BAR_HEIGHT)

                bars_html += f"""
                    <div style="
                        display:inline-block;
                        width:28px;
                        height:{bar_height}px;
                        background:#4C78A8;
                        margin-right:6px;
                        vertical-align:bottom;
                        border-radius:3px;
                    " title="{lbl}: {val}"></div>
                """

            popup_html = f"""
            <div style="
                font-family:Arial, sans-serif;
                padding:6px;
                width:330px;
            ">
                <h4 style="margin:0 0 6px;">Pump {pid}</h4>

                <div style="
                    padding:10px;
                    background:white;
                    border-radius:8px;
                    border:1px solid #ccc;
                    box-shadow:0 0 6px rgba(0,0,0,0.15);
                ">
                    <div style="white-space:nowrap; overflow-x:auto;">
                        {bars_html}
                    </div>

                    <div style="margin-top:8px; font-size:11px; color:#444;">
                        bins: {", ".join(labels)}
                    </div>
                </div>
            </div>
            """

            prow_wgs = pumps_wgs_reset.geometry.iloc[pid]

            folium.Marker(
                location=[prow_wgs.y, prow_wgs.x],
                icon=folium.Icon(color="darkred", icon="tint"),
                popup=folium.Popup(popup_html, max_width=360)
            ).add_to(pump_group)

        pump_group.add_to(m)
    # ------------------------------------------------------
    # SEWER LAYER
    # ------------------------------------------------------
    if show_sewer:

        sewer_group = folium.FeatureGroup(
            name="Sewer Grates", show=False
        )

        for _, row in sewer_wgs.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=3,
                color="green",
                fill=True,
                fill_opacity=0.7,
            ).add_to(sewer_group)

        sewer_group.add_to(m)

    # ------------------------------------------------------
    # MAP LEGEND
    # ------------------------------------------------------
    from branca.element import Element

    legend_html = """
    <div style="
        position: fixed;
        bottom: 60px;
        left: 10px;
        width: 240px;
        z-index:9999;
        font-size:12px;
        font-family: Arial, sans-serif;
    ">
      <div style="
          background:white;
          padding:10px;
          border-radius:6px;
          box-shadow:2px 2px 6px rgba(0,0,0,0.25)
      ">
        <b>Legend</b><br>
        <span style="display:inline-block;
                      width:12px;height:12px;
                      background:#222;margin-right:6px;
                      border-radius:3px"></span>
        Death points<br>

        <span style="display:inline-block;
                      width:12px;height:12px;
                      background:rgba(255,0,0,0.6);
                      margin-right:6px;border-radius:3px"></span>
        Pump histogram popup<br>

        <span style="display:inline-block;
                      width:12px;height:12px;
                      background:green;
                      margin-right:6px;border-radius:3px"></span>
        Sewer grates<br>
      </div>
    </div>
    """

    m.get_root().html.add_child(Element(legend_html))

    # ------------------------------------------------------
    # LAYER CONTROL
    # ------------------------------------------------------
    folium.LayerControl(collapsed=False).add_to(m)

    return m
# ---------------- Streamlit UI (PART 4/4) ----------------
st.set_page_config(page_title="John Snow — Cholera Dashboard", layout="wide")
st.title("John Snow 1854 — Cholera Map Dashboard")

st.sidebar.header("Controls & Map Options")

# Layer toggles
st.sidebar.subheader("Layers")
show_deaths   = st.sidebar.checkbox("Show Death points", value=True)
show_pumps    = st.sidebar.checkbox("Show Pumps (popup hist)", value=True)
show_sewer    = st.sidebar.checkbox("Show Sewer grates", value=False)
show_heat     = st.sidebar.checkbox("Show HeatMap", value=False)
show_clusters = st.sidebar.checkbox("Show DBSCAN clusters (overlay)", value=False)

# DBSCAN settings (used for stats & optional overlay)
st.sidebar.subheader("DBSCAN settings (meters)")
dbscan_eps = st.sidebar.slider("eps (meters)", 5, 200, 25, step=1)
dbscan_min = st.sidebar.slider("min_samples", 1, 10, 3, step=1)

# Other controls (bandwidth for future use)
st.sidebar.subheader("Other")
bandwidth = st.sidebar.slider("KDE bandwidth (not used in this build)", 10, 200, 50, step=5)

# Regenerate map (simple trigger)
if st.sidebar.button("Regenerate map"):
    st.session_state["regen"] = True

# Load data (already cached)
with st.spinner("Loading datasets..."):
    deaths, pumps, sewer = load_data()
    deaths_gdf, pumps_gdf, sewer_gdf = prepare_gdfs(deaths, pumps, sewer)

# Main layout: map + side stats
col1, col2 = st.columns((3,1))

with col1:
    st.subheader("Interactive Map")
    with st.spinner("Generating map..."):
        m = build_folium_map(
            deaths_gdf,
            pumps_gdf,
            sewer_gdf,
            show_deaths=show_deaths,
            show_pumps=show_pumps,
            show_sewer=show_sewer,
            show_heat=show_heat,
            bandwidth=bandwidth,
        )

        # If user wants DBSCAN clusters overlay (compute and add)
        if show_clusters:
            try:
                coords_proj = np.column_stack([
                    deaths_gdf.geometry.x.values,
                    deaths_gdf.geometry.y.values
                ])
                if len(coords_proj) > 0:
                    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min).fit(coords_proj)
                    deaths_gdf['cluster'] = db.labels_
                    # convert cluster centroids to WGS for markers
                    clusters = []
                    for lbl in sorted([l for l in np.unique(db.labels_) if l != -1]):
                        sub = deaths_gdf[deaths_gdf['cluster'] == lbl]
                        if not sub.empty:
                            centroid = sub.geometry.unary_union.centroid
                            centroid_wgs = gpd.GeoSeries([centroid], crs=deaths_gdf.crs).to_crs(epsg=4326).iloc[0]
                            folium.Marker(
                                location=[centroid_wgs.y, centroid_wgs.x],
                                popup=f"Cluster centroid {lbl}",
                                icon=folium.DivIcon(html=f"<div style='font-size:10px;color:black;background:rgba(255,255,255,0.85);padding:3px;border-radius:4px'>C{lbl}</div>")
                            ).add_to(m)
                    # add small circle markers for clustered points (colored)
                    cmap = plt.get_cmap("tab10")
                    for _, row in gpd.GeoDataFrame(deaths_gdf).to_crs(epsg=4326).iterrows():
                        lbl = int(row.get("cluster", -1))
                        if lbl != -1:
                            rgba = cmap(lbl % 10)
                            color = mcolors.to_hex(rgba[:3])
                            folium.CircleMarker(
                                location=[row.geometry.y, row.geometry.x],
                                radius=4,
                                color=color,
                                fill=True,
                                fill_opacity=0.9,
                                popup=f"Cluster {lbl}"
                            ).add_to(m)
                else:
                    st.warning("Not enough points for DBSCAN clustering.")
            except Exception as e:
                st.error(f"Error computing DBSCAN overlay: {e}")

    # Render folium map (st_folium expects map object)
    st_folium(m, width=None, height=700)

with col2:
    st.subheader("Quick statistics")
    st.write(f"Total death points: **{len(deaths_gdf)}**")
    st.write(f"Total pumps: **{len(pumps_gdf)}**")

    # clusters count (compute on-the-fly if available)
    if 'cluster' in deaths_gdf.columns:
        labels = deaths_gdf['cluster'].unique()
        num_clusters = int((labels >= 0).sum()) if labels.size else 0
        st.write(f"DBSCAN clusters (non-noise): **{num_clusters}**")
    else:
        st.write("DBSCAN clusters (non-noise): **not computed**")

    st.markdown("---")
    st.write("Download map HTML (final output if exists):")
    if FINAL_HTML_PATH.exists():
        with open(FINAL_HTML_PATH, "rb") as fh:
            data = fh.read()
            st.download_button("Download final_cholera_map.html", data, file_name=FINAL_HTML_PATH.name, mime="text/html")
    else:
        st.info("Final HTML not found. Use 'Save final map as HTML' to create one below.")

# ---------------- Export final map ----------------
st.markdown("---")
st.subheader("Export Final Map to HTML")
html_path = FINAL_HTML_PATH

if st.button("Save final map as HTML"):
    try:
        m.save(str(html_path))
        st.success(f"Final map saved to: {html_path}")
    except Exception as e:
        st.error(f"Error while saving final map: {e}")

if html_path.exists():
    with open(html_path, "rb") as fh:
        html_bytes = fh.read()
        st.download_button(
            label="📥 Download Final Cholera Map (HTML)",
            data=html_bytes,
            file_name=html_path.name,
            mime="text/html"
        )
else:
    st.info("Final HTML not found. Click 'Save final map as HTML' to create it.")

st.markdown("---")
st.caption("Created for GES723 John Snow Lab — interactive dashboard by Norliana Mokhtar.")
