"""
johnsnow_dashboard_app.py
"""

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

# ---------------- PATHS: use relative paths so Streamlit Cloud works ----------------
# Repo must contain a folder named exactly "data" (lowercase)
DATA_DIR = Path("data")
OUT_DIR  = Path("Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# default final html (if you previously generated it)
FINAL_HTML_PATH = OUT_DIR / "final_cholera_map.html"

# ---------------- Helpers: load & cache ----------------
@st.cache_data(show_spinner=False)
def load_csv_safe(fp: Path):
    # lightweight check, raise clear error if missing
    if not fp.exists():
        raise FileNotFoundError(f"Required file not found: {fp}")
    return pd.read_csv(fp)

@st.cache_data(show_spinner=False)
def load_data():
    # filenames expected in repo/data/
    deaths = load_csv_safe(DATA_DIR / "deaths_by_bldg.csv")
    pumps  = load_csv_safe(DATA_DIR / "pumps.csv")
    sewer  = load_csv_safe(DATA_DIR / "sewergrates_ventilators.csv")
    return deaths, pumps, sewer

# Create GeoDataFrames in projected CRS
@st.cache_data(show_spinner=False)
def prepare_gdfs(deaths, pumps, sewer, xcol="COORD_X", ycol="COORD_Y", crs_proj="EPSG:27700"):
    # make sure columns exist
    for df, name in [(deaths, "deaths"), (pumps, "pumps"), (sewer, "sewer")]:
        if xcol not in df.columns or ycol not in df.columns:
            raise KeyError(f"Expected columns '{xcol}' and '{ycol}' in {name} csv.")
    deaths_gdf = gpd.GeoDataFrame(deaths.copy(),
                                  geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]),
                                  crs=crs_proj)
    pumps_gdf = gpd.GeoDataFrame(pumps.copy(),
                                  geometry=gpd.points_from_xy(pumps[xcol], pumps[ycol]),
                                  crs=crs_proj)
    sewer_gdf = gpd.GeoDataFrame(sewer.copy(),
                                  geometry=gpd.points_from_xy(sewer[xcol], sewer[ycol]),
                                  crs=crs_proj)
    return deaths_gdf, pumps_gdf, sewer_gdf

# Build folium map dynamically based on control params
def build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf,
                     kde_n=400, kde_cmap="magma", kde_opacity=0.75,
                     heat_radius=18, heat_blur=20,
                     dbscan_eps=15, dbscan_min_samples=5,
                     show_heat=True, show_kde=True, show_clusters=True, show_pumps=True, show_sewer=False):
    # Projected CRS (EPSG:27700) assumed for dataset
    crs_proj = deaths_gdf.crs
    xs = deaths_gdf.geometry.x.values
    ys = deaths_gdf.geometry.y.values
    coords = np.vstack([xs, ys])
    kde_vals = gaussian_kde(coords)

    xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
    # grid
    xx, yy = np.mgrid[xmin:xmax:kde_n*1j, ymin:ymax:kde_n*1j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde_vals(grid_coords).reshape(xx.shape)

    # Save temporary KDE raster (transparent)
    fig, ax = plt.subplots(figsize=(8,8), dpi=150)
    ax.imshow(np.rot90(zz), cmap=kde_cmap, extent=[xmin, xmax, ymin, ymax], alpha=1.0)
    ax.axis('off')
    plt.tight_layout(pad=0)
    tmp_kde_png = OUT_DIR / "tmp_kde.png"
    fig.savefig(tmp_kde_png, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # make dark background transparent
    img_pil = Image.open(tmp_kde_png).convert("RGBA")
    arr = np.array(img_pil)
    threshold = 20
    mask = (arr[:,:,0] < threshold) & (arr[:,:,1] < threshold) & (arr[:,:,2] < threshold)
    arr[mask,3] = 0
    img_pil2 = Image.fromarray(arr)
    kde_png_trans = OUT_DIR / "tmp_kde_trans.png"
    img_pil2.save(kde_png_trans)

    # Convert bounds to WGS84
    corners = gpd.GeoSeries([Point(xmin, ymin), Point(xmax, ymax)], crs=crs_proj).to_crs(epsg=4326)
    min_lon = float(corners.geometry.x.min()); max_lon = float(corners.geometry.x.max())
    min_lat = float(corners.geometry.y.min()); max_lat = float(corners.geometry.y.max())
    overlay_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Prepare WGS points
    deaths_wgs = deaths_gdf.to_crs(epsg=4326)
    pumps_wgs = pumps_gdf.to_crs(epsg=4326)
    sewer_wgs = sewer_gdf.to_crs(epsg=4326)

    # Heat data
    heat_data = [[row.geometry.y, row.geometry.x, 1] for _, row in deaths_wgs.iterrows()]

    # Prepare map center
    center = [float(deaths_wgs.geometry.y.mean()), float(deaths_wgs.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")

    # Add KDE raster overlay if desired
    if show_kde:
        folium.raster_layers.ImageOverlay(
            name="KDE raster (smooth)",
            image=str(kde_png_trans),
            bounds=overlay_bounds,
            opacity=kde_opacity,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(m)

    # Add HeatMap (point-weighted)
    if show_heat:
        heat_layer = folium.FeatureGroup(name="HeatMap (point-weighted)", show=False)
        HeatMap(heat_data, radius=heat_radius, blur=heat_blur, max_zoom=18).add_to(heat_layer)
        heat_layer.add_to(m)

    # DBSCAN clustering (projected CRS)
    if show_clusters:
        coords_proj = np.column_stack([deaths_gdf.geometry.x, deaths_gdf.geometry.y])
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_proj)
        deaths_gdf['cluster'] = db.labels_
        deaths_wgs['cluster'] = deaths_gdf['cluster'].values
        unique_labels = sorted([l for l in np.unique(db.labels_) if l != -1])
        cmap = plt.get_cmap("tab10")
        for _, row in deaths_wgs.iterrows():
            lbl = int(row.get("cluster", -1))
            if lbl != -1:
                color = mcolors.to_hex(cmap(lbl % 10)[:3])
                folium.CircleMarker(location=[row.geometry.y, row.geometry.x],
                                    radius=5, color=color, fill=True, fill_opacity=0.85,
                                    popup=f"Cluster: {lbl}").add_to(m)
        # centroids
        for lbl in unique_labels:
            sub = deaths_gdf[deaths_gdf['cluster'] == lbl]
            if not sub.empty:
                centroid = sub.geometry.unary_union.centroid
                centroid_wgs = gpd.GeoSeries([centroid], crs=crs_proj).to_crs(epsg=4326).iloc[0]
                folium.Marker(location=[centroid_wgs.y, centroid_wgs.x],
                              popup=f"Cluster centroid {lbl}",
                              icon=folium.DivIcon(html=f"<div style='font-size:10px;color:black;background:rgba(255,255,255,0.85);padding:3px;border-radius:4px'>C{lbl}</div>")).add_to(m)
    else:
        # ensure column exists even if clustering not shown
        deaths_gdf['cluster'] = -1
        deaths_wgs['cluster'] = -1

    # Pumps layer + popup chart (simple inline bars)
    if show_pumps:
        pump_group = folium.FeatureGroup(name="Pumps (click for distance chart)", show=True)
        pumps_iter = pumps_gdf.reset_index(drop=True)
        pumps_wgs_reset = pumps_wgs.reset_index(drop=True)
        for pid, prow in pumps_iter.iterrows():
            dists = deaths_gdf.geometry.distance(prow.geometry)
            bins = [0, 10, 25, 50, 100, 200, np.inf]
            labels = ["0-10","10-25","25-50","50-100","100-200",">200"]
            binned = pd.cut(dists, bins=bins, labels=labels)
            summary = binned.value_counts().reindex(labels).fillna(0).astype(int)
            bars = "".join([f"<div style='display:inline-block;width:30px;height:{max(8, int(v*6))}px;background:#4C78A8;margin-right:3px;vertical-align:bottom' title='{labels[i]}: {v}'></div>"
                            for i,v in enumerate(summary.values)])
            html = f"<div style='width:320px'><b>Pump {pid}</b><br>{bars}<br><small>bins: {', '.join(labels)}</small></div>"
            prow_wgs = pumps_wgs_reset.geometry.iloc[pid]
            folium.Marker(location=[prow_wgs.y, prow_wgs.x],
                          popup=folium.Popup(html, max_width=360),
                          icon=folium.Icon(color="darkred", icon="tint")).add_to(pump_group)
        pump_group.add_to(m)

    # Sewer layer
    if show_sewer:
        sewer_group = folium.FeatureGroup(name="Sewer grates", show=False)
        for _, row in sewer_wgs.iterrows():
            folium.CircleMarker([row.geometry.y, row.geometry.x],
                                radius=3, color="green", fill=True, fill_opacity=0.6).add_to(sewer_group)
        sewer_group.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Title + legend
    title_html = """
         <h3 align="center" style="font-size:18px">
             <b>Cholera Outbreak Analysis — Density, Clusters & Distance Patterns</b>
         </h3>
         """
    from branca.element import Element
    m.get_root().html.add_child(Element(title_html))

    legend_html = """
    <div style="position: fixed; bottom: 60px; left: 10px; width: 240px; z-index:9999; font-size:12px;">
      <div style="background:white; padding:10px; border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.25)">
        <b>Legend</b><br>
        <span style="display:inline-block;width:12px;height:12px;background:rgba(180,30,20,0.75);margin-right:6px;border-radius:3px"></span> KDE hotspot<br>
        <span style="display:inline-block;width:10px;height:10px;background:#000;margin-right:6px;border-radius:2px"></span> Death points<br>
        <svg width="12" height="12" style="vertical-align:middle;margin-right:6px;"><polygon points="6,0 12,12 0,12" style="fill:darkred"/></svg> Pumps<br>
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    return m

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="John Snow — Cholera Dashboard", layout="wide")
st.title("John Snow 1854 — Cholera Map Dashboard")

st.sidebar.header("Controls & Map Options")

# Load data
with st.spinner("Loading datasets..."):
    deaths, pumps, sewer = load_data()
    deaths_gdf, pumps_gdf, sewer_gdf = prepare_gdfs(deaths, pumps, sewer)

# Sidebar: layer toggles
st.sidebar.subheader("Layer toggles")
show_kde     = st.sidebar.checkbox("Show KDE raster", value=True)
show_heat    = st.sidebar.checkbox("Show point HeatMap", value=False)
show_clusters= st.sidebar.checkbox("Show DBSCAN clusters", value=True)
show_pumps   = st.sidebar.checkbox("Show Pumps (with popup charts)", value=True)
show_sewer   = st.sidebar.checkbox("Show Sewer grates", value=False)

# KDE controls
st.sidebar.subheader("KDE settings")
kde_n = st.sidebar.slider("KDE resolution (n)", 200, 900, 400, step=50)
kde_opacity = st.sidebar.slider("KDE opacity", 0.0, 1.0, 0.75, step=0.05)
kde_cmap = st.sidebar.selectbox("KDE colormap", ["magma","inferno","plasma","viridis"], index=0)

# Heat settings
st.sidebar.subheader("HeatMap settings")
heat_radius = st.sidebar.slider("Heat radius", 6, 40, 18, step=2)
heat_blur = st.sidebar.slider("Heat blur", 0, 40, 20, step=2)

# DBSCAN settings
st.sidebar.subheader("DBSCAN settings (meters)")
dbscan_eps = st.sidebar.slider("eps (meters)", 5, 100, 15, step=1)
dbscan_min = st.sidebar.slider("min_samples", 2, 10, 5, step=1)

# Buttons
if st.sidebar.button("Regenerate map"):
    st.session_state["regen"] = True

# Main layout: left map, right controls & stats
col1, col2 = st.columns((3,1))

with col1:
    st.subheader("Interactive Map")
    with st.spinner("Generating map..."):
        m = build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf,
                             kde_n=kde_n, kde_cmap=kde_cmap, kde_opacity=kde_opacity,
                             heat_radius=heat_radius, heat_blur=heat_blur,
                             dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min,
                             show_heat=show_heat, show_kde=show_kde, show_clusters=show_clusters,
                             show_pumps=show_pumps, show_sewer=show_sewer)
    # Render folium map
    st_folium(m, width="100%", height=700)

with col2:
    st.subheader("Quick statistics")
    st.write(f"Total death points: **{len(deaths_gdf)}**")
    st.write(f"Total pumps: **{len(pumps_gdf)}**")
    st.write("DBSCAN clusters (non-noise):")
    try:
        labels = deaths_gdf['cluster'].unique()
        num_clusters = int((labels >= 0).sum()) if labels.size else 0
        st.write(f"- Number of clusters (excluding noise): **{num_clusters}**")
    except Exception:
        st.write("- not computed")

    st.markdown("---")
    st.write("Download map HTML (final polished output if exists):")
    if FINAL_HTML_PATH.exists():
        with open(FINAL_HTML_PATH, "rb") as fh:
            data = fh.read()
            st.download_button("Download final_cholera_map_polished.html", data, file_name=FINAL_HTML_PATH.name, mime="text/html")
    else:
        st.info("Final HTML not found. Use 'Regenerate map' to produce a fresh HTML file from this dashboard (feature coming).")

st.markdown("---")
st.caption("Created for GES723 John Snow Lab — interactive dashboard by Norliana Mokhtar.")
