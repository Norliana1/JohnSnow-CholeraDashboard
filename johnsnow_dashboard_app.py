# johnsnow_dashboard_app.py

import io
import os
from pathlib import Path
import base64
import json
import tempfile
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, mapping
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
    # expected files: deaths_by_bldg.csv, pumps.csv, sewergrates_ventilators.csv
    deaths = load_csv_safe(DATA_DIR / "deaths_by_bldg.csv")
    pumps  = load_csv_safe(DATA_DIR / "pumps.csv")
    sewer  = load_csv_safe(DATA_DIR / "sewergrates_ventilators.csv")
    return deaths, pumps, sewer

@st.cache_data(show_spinner=False)
def prepare_gdfs(deaths, pumps, sewer, xcol="COORD_X", ycol="COORD_Y", crs_proj="EPSG:27700"):
    # Validate and coerce
    for df, name in [(deaths, "deaths"), (pumps, "pumps"), (sewer, "sewer")]:
        if xcol not in df.columns or ycol not in df.columns:
            raise KeyError(f"Expected columns '{xcol}' and '{ycol}' in {name} csv.")
    deaths = deaths.copy(); pumps = pumps.copy(); sewer = sewer.copy()
    deaths[xcol] = pd.to_numeric(deaths[xcol], errors="coerce"); deaths[ycol] = pd.to_numeric(deaths[ycol], errors="coerce")
    pumps[xcol]  = pd.to_numeric(pumps[xcol], errors="coerce"); pumps[ycol]  = pd.to_numeric(pumps[ycol], errors="coerce")
    sewer[xcol]  = pd.to_numeric(sewer[xcol], errors="coerce"); sewer[ycol]  = pd.to_numeric(sewer[ycol], errors="coerce")
    deaths = deaths.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    pumps  = pumps.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    sewer  = sewer.dropna(subset=[xcol, ycol]).reset_index(drop=True)

    deaths_gdf = gpd.GeoDataFrame(deaths, geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]), crs=crs_proj)
    pumps_gdf  = gpd.GeoDataFrame(pumps,  geometry=gpd.points_from_xy(pumps[xcol], pumps[ycol]), crs=crs_proj)
    sewer_gdf  = gpd.GeoDataFrame(sewer,  geometry=gpd.points_from_xy(sewer[xcol], sewer[ycol]), crs=crs_proj)
    return deaths_gdf, pumps_gdf, sewer_gdf

# Utility: convert fig to base64
def fig_to_base64(fig, fmt="png", dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------- Map generation (cached) ----------------
def generate_kde_image(deaths_gdf, kde_n=300, kde_cmap="magma", threshold=20, tmpname="kde_temp.png"):
    # produce transparent KDE PNG (projected CRS)
    xs = deaths_gdf.geometry.x.values
    ys = deaths_gdf.geometry.y.values
    coords = np.vstack([xs, ys])
    kde_vals = gaussian_kde(coords)
    xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
    xx, yy = np.mgrid[xmin:xmax:kde_n*1j, ymin:ymax:kde_n*1j]
    zz = kde_vals(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8,8), dpi=150)
    ax.imshow(np.rot90(zz), cmap=kde_cmap, extent=[xmin, xmax, ymin, ymax], alpha=1.0)
    ax.axis('off')
    plt.tight_layout(pad=0)
    tmp_kde_png = OUT_DIR / tmpname
    fig.savefig(tmp_kde_png, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # make dark pixels transparent
    img_pil = Image.open(tmp_kde_png).convert("RGBA")
    arr = np.array(img_pil)
    mask = (arr[:,:,0] < threshold) & (arr[:,:,1] < threshold) & (arr[:,:,2] < threshold)
    arr[mask,3] = 0
    img_pil2 = Image.fromarray(arr)
    tmp_kde_png_trans = OUT_DIR / (tmpname.replace(".png", "_trans.png"))
    img_pil2.save(tmp_kde_png_trans)
    return str(tmp_kde_png_trans)

def compute_dbscan(deaths_gdf, eps=15, min_samples=5):
    coords_proj = np.column_stack([deaths_gdf.geometry.x, deaths_gdf.geometry.y])
    if len(coords_proj) == 0:
        deaths_gdf['cluster'] = -1
        return deaths_gdf, []
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_proj)
    deaths_gdf = deaths_gdf.copy()
    deaths_gdf['cluster'] = db.labels_
    unique_labels = sorted([l for l in np.unique(db.labels_) if l != -1])
    return deaths_gdf, unique_labels

def build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf,
                     kde_png_trans=None, kde_opacity=0.75,
                     heat_radius=18, heat_blur=20,
                     show_kde=True, show_heat=False, show_clusters=True, show_pumps=True, show_sewer=False,
                     dbscan_eps=15, dbscan_min_samples=5,
                     pump_radius_m=100, show_buffers=False):
    """
    Build and return folium.Map object (WGS84)
    pump_radius_m: buffer radius in meters for pump influence
    show_buffers: whether to draw pump buffers as GeoJSON
    """
    # make local copies
    deaths_gdf = deaths_gdf.copy()
    pumps_gdf = pumps_gdf.copy()
    sewer_gdf = sewer_gdf.copy()
    crs_proj = deaths_gdf.crs

    # convert bounding box & WGS points
    deaths_wgs = deaths_gdf.to_crs(epsg=4326)
    pumps_wgs = pumps_gdf.to_crs(epsg=4326)
    sewer_wgs = sewer_gdf.to_crs(epsg=4326)

    # center map
    center = [float(deaths_wgs.geometry.y.mean()), float(deaths_wgs.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")

    # Add KDE overlay (projected->WGS bounds computed from projected bbox)
    if kde_png_trans is not None and show_kde:
        xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
        corners = gpd.GeoSeries([Point(xmin, ymin), Point(xmax, ymax)], crs=crs_proj).to_crs(epsg=4326)
        min_lon = float(corners.geometry.x.min()); max_lon = float(corners.geometry.x.max())
        min_lat = float(corners.geometry.y.min()); max_lat = float(corners.geometry.y.max())
        overlay_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        folium.raster_layers.ImageOverlay(
            name="KDE raster",
            image=str(kde_png_trans),
            bounds=overlay_bounds,
            opacity=kde_opacity,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(m)

    # Heatmap
    if show_heat:
        heat_data = [[row.geometry.y, row.geometry.x, 1] for _, row in deaths_wgs.iterrows()]
        HeatMap(heat_data, radius=heat_radius, blur=heat_blur, max_zoom=18).add_to(
            folium.FeatureGroup(name="Heatmap", show=False).add_to(m)
        )

    # DBSCAN clusters
    if show_clusters:
        deaths_clustered, unique_labels = compute_dbscan(deaths_gdf, eps=dbscan_eps, min_samples=dbscan_min_samples)
        deaths_wgs['cluster'] = deaths_clustered['cluster'].values
        cmap = plt.get_cmap("tab10")
        for _, row in deaths_wgs.iterrows():
            lbl = int(row.get("cluster", -1))
            if lbl != -1:
                rgba = cmap(lbl % 10)
                color = mcolors.to_hex(rgba[:3])
                folium.CircleMarker(location=[row.geometry.y, row.geometry.x],
                                    radius=4, color=color, fill=True, fill_opacity=0.85,
                                    popup=f"Cluster: {lbl}").add_to(m)
        # centroids
        for lbl in unique_labels:
            sub = deaths_gdf[deaths_gdf['cluster'] == lbl]
            if not sub.empty:
                try:
                    centroid = sub.geometry.unary_union.centroid
                except Exception:
                    centroid = sub.geometry.centroid.iloc[0]
                centroid_wgs = gpd.GeoSeries([centroid], crs=crs_proj).to_crs(epsg=4326).iloc[0]
                folium.Marker(location=[centroid_wgs.y, centroid_wgs.x],
                              popup=f"Cluster centroid {lbl}",
                              icon=folium.DivIcon(html=f"<div style='font-size:10px;color:black;background:rgba(255,255,255,0.85);padding:3px;border-radius:4px'>C{lbl}</div>")).add_to(m)
    else:
        deaths_wgs['cluster'] = -1

    # Pumps & pump buffers
    if show_pumps:
        pump_group = folium.FeatureGroup(name="Pumps (click for details)", show=True)
        pumps_wgs_reset = pumps_wgs.reset_index(drop=True)
        pumps_proj_reset = pumps_gdf.reset_index(drop=True)
        for pid, prow_wgs in pumps_wgs_reset.iterrows():
            popup_html = f"<b>Pump {pid}</b><br>"
            # compute counts within radius in projected CRS for accuracy
            pump_proj_geom = pumps_proj_reset.geometry.iloc[pid]
            dists = deaths_gdf.geometry.distance(pump_proj_geom)  # in projection units (meters if EPSG:27700)
            in_radius = (dists <= pump_radius_m).sum()
            popup_html += f"Deaths within {pump_radius_m} m: <b>{int(in_radius)}</b><br>"
            folium.Marker(location=[prow_wgs.geometry.y, prow_wgs.geometry.x],
                          popup=folium.Popup(popup_html, max_width=320),
                          icon=folium.Icon(color="darkred", icon="tint")).add_to(pump_group)
            # draw buffer if requested
            if show_buffers:
                buff = pump_proj_geom.buffer(pump_radius_m)
                buff_wgs = gpd.GeoSeries([buff], crs=deaths_gdf.crs).to_crs(epsg=4326).__geo_interface__
                folium.GeoJson(buff_wgs, name=f"Pump_{pid}_buffer", style_function=lambda x: {"fillColor":"#ffcccc","color":"#ff4444","weight":1,"fillOpacity":0.25}).add_to(pump_group)
        pump_group.add_to(m)

    # Sewer
    if show_sewer:
        sewer_group = folium.FeatureGroup(name="Sewer grates", show=False)
        for _, row in sewer_wgs.iterrows():
            folium.CircleMarker([row.geometry.y, row.geometry.x],
                                radius=3, color="green", fill=True, fill_opacity=0.6).add_to(sewer_group)
        sewer_group.add_to(m)

    # All death points cluster for exploration (small black dots)
    mc = MarkerCluster(name="All death points (clustered)").add_to(m)
    for _, row in deaths_wgs.iterrows():
        popup = ""
        if "ID" in row:
            popup += f"ID: {row['ID']}<br>"
        popup += f"Cluster: {row.get('cluster', '')}"
        folium.CircleMarker([row.geometry.y, row.geometry.x],
                            radius=3, color="black", fill=True, fill_opacity=0.7,
                            popup=popup).add_to(mc)

    # title + legend
    title_html = """
         <h3 align="center" style="font-size:18px"><b>Cholera Outbreak Analysis — Density, Clusters & Pump Influence</b></h3>
         """
    from branca.element import Element
    m.get_root().html.add_child(Element(title_html))

    legend_html = """
    <div style="position: fixed; bottom: 60px; left: 10px; width: 260px; z-index:9999; font-size:12px;">
      <div style="background:white; padding:10px; border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.25)">
        <b>Legend</b><br>
        <span style="display:inline-block;width:12px;height:12px;background:rgba(180,30,20,0.75);margin-right:6px;border-radius:3px"></span> KDE hotspot<br>
        <span style="display:inline-block;width:10px;height:10px;background:#000;margin-right:6px;border-radius:2px"></span> Death points<br>
        <svg width="12" height="12" style="vertical-align:middle;margin-right:6px;"><polygon points="6,0 12,12 0,12" style="fill:darkred"/></svg> Pumps<br>
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m, deaths_wgs, pumps_wgs

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="John Snow — Cholera Premium Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🗺️ John Snow 1854 — Cholera Dashboard (Premium)")

# Top-level description / branding
with st.expander("About this dashboard (click)"):
    st.markdown("""
    **Interactive spatial analysis of John Snow's 1854 cholera dataset.**
    - Use the sidebar to tune KDE, DBSCAN, heatmap and pump influence radius.
    - Download outputs and CSV summaries from the Export panel.
    - Created by **Norliana Mokhtar** for GES723 John Snow Lab.
    """)

# Theme toggle (affects only explanatory text; Streamlit theme must be set in .streamlit/config.toml for full effect)
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

# Load data (cached)
with st.spinner("Loading datasets..."):
    deaths_df, pumps_df, sewer_df = load_data()
    deaths_gdf, pumps_gdf, sewer_gdf = prepare_gdfs(deaths_df, pumps_df, sewer_df)

# Sidebar controls
st.sidebar.header("Map controls & analysis")

# Layer toggles
st.sidebar.subheader("Layers")
show_kde     = st.sidebar.checkbox("Show KDE raster", value=True)
show_heat    = st.sidebar.checkbox("Show point HeatMap", value=False)
show_clusters= st.sidebar.checkbox("Show DBSCAN clusters", value=True)
show_pumps   = st.sidebar.checkbox("Show Pumps (with popup charts)", value=True)
show_sewer   = st.sidebar.checkbox("Show Sewer grates", value=False)

# KDE settings
st.sidebar.subheader("KDE settings")
kde_n = st.sidebar.slider("KDE resolution (n)", 150, 800, 300, step=50)
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

# Pump influence
st.sidebar.subheader("Pump influence")
pump_radius_m = st.sidebar.slider("Pump radius (meters)", 10, 500, 100, step=10)
show_buffers = st.sidebar.checkbox("Show pump buffers on map", value=False)

# Regenerate / quick actions
if st.sidebar.button("Regenerate map / re-run analysis"):
    # bump cache key by writing a small flag in session state
    st.experimental_rerun()

# ---------------- Build assets ----------------
with st.spinner("Preparing KDE image..."):
    kde_png_trans = generate_kde_image(deaths_gdf, kde_n=kde_n, kde_cmap=kde_cmap)

with st.spinner("Generating interactive map..."):
    m, deaths_wgs, pumps_wgs = build_folium_map(
        deaths_gdf, pumps_gdf, sewer_gdf,
        kde_png_trans=kde_png_trans, kde_opacity=kde_opacity,
        heat_radius=heat_radius, heat_blur=heat_blur,
        show_kde=show_kde, show_heat=show_heat, show_clusters=show_clusters,
        show_pumps=show_pumps, show_sewer=show_sewer,
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min,
        pump_radius_m=pump_radius_m, show_buffers=show_buffers
    )

# ---------------- Layout: Map + Right panel ----------------
col1, col2 = st.columns((3,1))
with col1:
    st.subheader("Interactive Map")
    st.info("Tip: Use the layer control on the map to toggle layers. Use sidebar to tune parameters.")
    st_folium(m, width=None, height=720, returned_objects=[])

with col2:
    st.subheader("Quick statistics & Analysis")
    total_deaths = len(deaths_gdf)
    total_pumps = len(pumps_gdf)
    total_sewer = len(sewer_gdf)
    st.markdown(f"- **Total death points:** {total_deaths}")
    st.markdown(f"- **Total pumps:** {total_pumps}")
    st.markdown(f"- **Total sewer grates:** {total_sewer}")

    # DBSCAN summary
    deaths_clustered, unique_labels = compute_dbscan(deaths_gdf, eps=dbscan_eps, min_samples=dbscan_min)
    labels = deaths_clustered['cluster'].unique() if 'cluster' in deaths_clustered.columns else np.array([-1])
    num_clusters = int((labels >= 0).sum()) if labels.size else 0
    st.markdown(f"- **DBSCAN clusters (non-noise):** {num_clusters}")

    if num_clusters > 0:
        st.write("**Cluster breakdown (size)**")
        cluster_counts = deaths_clustered[deaths_clustered["cluster"] >= 0]["cluster"].value_counts().sort_index()
        for cid, count in cluster_counts.items():
            st.write(f"- Cluster {cid}: **{int(count)}** deaths")

    # Pump influence table (counts within radius)
    st.markdown("**Pump influence (deaths within radius)**")
    pump_counts = []
    for pid, prow in pumps_gdf.reset_index(drop=True).iterrows():
        dists = deaths_gdf.geometry.distance(prow.geometry)
        cnt = int((dists <= pump_radius_m).sum())
        pump_counts.append({"pump_id": int(pid), "count_within_radius": cnt})
    pump_counts_df = pd.DataFrame(pump_counts)
    st.dataframe(pump_counts_df)

    # Distance summary (min/mean/median) from deaths to nearest pump
    # compute nearest distances in projected CRS (meters)
    if len(pumps_gdf) > 0:
        # build numpy arrays for speed
        pump_coords = np.vstack([pumps_gdf.geometry.x, pumps_gdf.geometry.y]).T
        death_coords = np.vstack([deaths_gdf.geometry.x, deaths_gdf.geometry.y]).T
        # compute distances matrix in projected CRS
        # for memory safety compute row-wise
        min_dists = []
        for dx, dy in death_coords:
            dd = np.sqrt((pump_coords[:,0]-dx)**2 + (pump_coords[:,1]-dy)**2)
            min_dists.append(dd.min())
        min_dists = np.array(min_dists)
        dist_stats = {"min_m": float(min_dists.min()), "mean_m": float(min_dists.mean()), "median_m": float(np.median(min_dists))}
        st.markdown("**Nearest pump distances (meters)**")
        st.write(dist_stats)
    else:
        st.info("No pumps found in dataset.")

# ---------------- Export / Downloads ----------------
st.markdown("---")
st.subheader("Export & Downloads")

# Save final HTML button
if st.button("Save final map as HTML"):
    try:
        m.save(str(FINAL_HTML_PATH))
        st.success(f"Saved HTML map to {FINAL_HTML_PATH}")
    except Exception as e:
        st.error(f"Error saving HTML: {e}")

# Download HTML if exists
if FINAL_HTML_PATH.exists():
    with open(FINAL_HTML_PATH, "rb") as fh:
        st.download_button("📥 Download HTML map", data=fh, file_name=FINAL_HTML_PATH.name, mime="text/html")

# Download cluster CSV
cluster_csv_buf = io.BytesIO()
cluster_df = deaths_clustered.copy()
if 'cluster' not in cluster_df.columns:
    cluster_df['cluster'] = -1
cluster_df.to_csv(cluster_csv_buf, index=False)
cluster_csv_buf.seek(0)
st.download_button("📥 Download death points + cluster (CSV)", data=cluster_csv_buf, file_name="deaths_with_cluster.csv", mime="text/csv")

# Download pump counts CSV
pump_csv_buf = io.BytesIO()
pump_counts_df.to_csv(pump_csv_buf, index=False)
pump_csv_buf.seek(0)
st.download_button("📥 Download pump influence counts (CSV)", data=pump_csv_buf, file_name="pump_influence_counts.csv", mime="text/csv")

# Quick report (simple text-based)
report_lines = []
report_lines.append(f"John Snow Cholera Dashboard — Report")
report_lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
report_lines.append(f"Total death points: {total_deaths}")
report_lines.append(f"Total pumps: {total_pumps}")
report_lines.append(f"Total sewer grates: {total_sewer}")
report_lines.append(f"DBSCAN clusters (non-noise): {num_clusters}")
report_lines.append(f"Pump radius used (m): {pump_radius_m}")
report_txt = "\n".join(report_lines).encode("utf-8")
st.download_button("📄 Download quick report (TXT)", data=report_txt, file_name="choleradashboard_report.txt", mime="text/plain")

st.caption("If you want an auto-generated PDF report (formatted), I can add that next (requires additional dependency like reportlab or weasyprint).")
