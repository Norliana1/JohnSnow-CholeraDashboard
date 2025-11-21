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
    # coerce to numeric
    deaths = deaths.copy()
    pumps  = pumps.copy()
    sewer  = sewer.copy()
    deaths[xcol] = pd.to_numeric(deaths[xcol], errors="coerce")
    deaths[ycol] = pd.to_numeric(deaths[ycol], errors="coerce")
    pumps[xcol]  = pd.to_numeric(pumps[xcol], errors="coerce")
    pumps[ycol]  = pd.to_numeric(pumps[ycol], errors="coerce")
    sewer[xcol]  = pd.to_numeric(sewer[xcol], errors="coerce")
    sewer[ycol]  = pd.to_numeric(sewer[ycol], errors="coerce")

    # drop rows without coords
    deaths = deaths.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    pumps  = pumps.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    sewer  = sewer.dropna(subset=[xcol, ycol]).reset_index(drop=True)

    deaths_gdf = gpd.GeoDataFrame(deaths,
                                  geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]),
                                  crs=crs_proj)
    pumps_gdf = gpd.GeoDataFrame(pumps,
                                  geometry=gpd.points_from_xy(pumps[xcol], pumps[ycol]),
                                  crs=crs_proj)
    sewer_gdf = gpd.GeoDataFrame(sewer,
                                  geometry=gpd.points_from_xy(sewer[xcol], sewer[ycol]),
                                  crs=crs_proj)
    return deaths_gdf, pumps_gdf, sewer_gdf

# ---------------- Build folium map ----------------
def build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf,
                     kde_n=400, kde_cmap="magma", kde_opacity=0.75,
                     heat_radius=18, heat_blur=20,
                     dbscan_eps=15, dbscan_min_samples=5,
                     show_heat=True, show_kde=True, show_clusters=True, show_pumps=True, show_sewer=False,
                     base_map="OpenStreetMap"):

    deaths_gdf = deaths_gdf.copy()
    pumps_gdf = pumps_gdf.copy()
    sewer_gdf = sewer_gdf.copy()

    crs_proj = deaths_gdf.crs
    xs = deaths_gdf.geometry.x.values
    ys = deaths_gdf.geometry.y.values
    coords = np.vstack([xs, ys])
    kde_vals = gaussian_kde(coords)

    xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
    xx, yy = np.mgrid[xmin:xmax:kde_n*1j, ymin:ymax:kde_n*1j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde_vals(grid_coords).reshape(xx.shape)

    # Create the base map with user-selected base map
    m = folium.Map(location=[float(deaths_gdf.geometry.y.mean()), float(deaths_gdf.geometry.x.mean())], 
                   zoom_start=16, tiles=base_map)

    # Add KDE
    if show_kde:
        folium.raster_layers.ImageOverlay(
            name="KDE raster (smooth)",
            image=str(kde_png_trans),
            bounds=[[ymin, xmin], [ymax, xmax]],
            opacity=kde_opacity,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(m)

    # Add Heatmap
    if show_heat:
        heat_data = [[row.geometry.y, row.geometry.x, 1] for _, row in deaths_gdf.iterrows()]
        HeatMap(heat_data, radius=heat_radius, blur=heat_blur, max_zoom=18).add_to(m)

    # Add Pumps with numbers and popup
    if show_pumps:
        for pid, prow in pumps_gdf.iterrows():
            folium.Marker(
                location=[prow.geometry.y, prow.geometry.x],
                icon=folium.Icon(color="darkred", icon="tint"),
                popup=f"Pump {pid}"
            ).add_to(m)

    # DBSCAN clusters
    coords_proj = np.column_stack([deaths_gdf.geometry.x, deaths_gdf.geometry.y])
    if show_clusters:
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_proj)
        deaths_gdf['cluster'] = db.labels_
        deaths_wgs['cluster'] = deaths_gdf['cluster'].values
        unique_labels = sorted([l for l in np.unique(db.labels_) if l != -1])
        cmap = plt.get_cmap("tab10")
        for _, row in deaths_gdf.iterrows():
            lbl = int(row.get("cluster", -1))
            if lbl != -1:
                rgba = cmap(lbl % 10)
                color = mcolors.to_hex(rgba[:3])
                folium.CircleMarker(location=[row.geometry.y, row.geometry.x],
                                    radius=5, color=color, fill=True, fill_opacity=0.85,
                                    popup=f"Cluster: {lbl}").add_to(m)
        for lbl in unique_labels:
            sub = deaths_gdf[deaths_gdf['cluster'] == lbl]
            if not sub.empty:
                centroid = sub.geometry.unary_union.centroid
                centroid_wgs = gpd.GeoSeries([centroid], crs=crs_proj).to_crs(epsg=4326).iloc[0]
                folium.Marker(location=[centroid_wgs.y, centroid_wgs.x],
                              popup=f"Cluster centroid {lbl}",
                              icon=folium.DivIcon(html=f"<div style='font-size:10px;color:black;"
                                                       f"background:rgba(255,255,255,0.85);padding:3px;"
                                                       f"border-radius:4px'>C{lbl}</div>")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="John Snow — Cholera Dashboard", layout="wide")
st.title("John Snow 1854 — Cholera Map Dashboard")

st.sidebar.header("Controls & Map Options")

# Sidebar Options
show_heat    = st.sidebar.checkbox("Show HeatMap", value=True)
show_kde     = st.sidebar.checkbox("Show KDE raster", value=True)
show_clusters= st.sidebar.checkbox("Show DBSCAN clusters", value=True)
show_pumps   = st.sidebar.checkbox("Show Pumps", value=True)
show_sewer   = st.sidebar.checkbox("Show Sewer Grates", value=False)
base_map     = st.sidebar.selectbox("Select BaseMap", ["OpenStreetMap", "Stamen Terrain", "CartoDB Positron", "Stamen Toner"])

# Load Data
deaths, pumps, sewer = load_data()
deaths_gdf, pumps_gdf, sewer_gdf = prepare_gdfs(deaths, pumps, sewer)

# Build Map
m = build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf, show_heat=show_heat, show_kde=show_kde, 
                     show_clusters=show_clusters, show_pumps=show_pumps, show_sewer=show_sewer, 
                     base_map=base_map)

# Display Map
st_folium(m, width=None, height=700)

# Quick Stats
st.subheader("Quick statistics")
st.write(f"Total death points: **{len(deaths_gdf)}**")
st.write(f"Total pumps: **{len(pumps_gdf)}**")
st.write(f"DBSCAN clusters (non-noise): **{len(np.unique(deaths_gdf['cluster']))}**")

# Save Final Map
st.subheader("Save Final Map")
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
        st.download_button(label="📥 Download Final Cholera Map (HTML)", data=html_bytes, 
                           file_name=html_path.name, mime="text/html")

st.markdown("---")
st.caption("Created for GES723 John Snow Lab — interactive dashboard by Norliana Mokhtar.")
