"""
johnsnow_dashboard_app.py
Advanced Streamlit dashboard for John Snow 1854 Cholera Map 
Features:
 - Dynamic Folium map with KDE raster overlay, point HeatMap, DBSCAN clusters
 - Sidebar controls for KDE opacity, heat radius/blur, DBSCAN eps/min_samples
 - Save current map as HTML (Outputs/final_cholera_map.html)
 - Export static high-resolution PNG (Outputs/final_cholera_map.png)
 - Download buttons for generated files

Usage:
 1. Place this file in your project folder (same level as Data and Outputs).
 2. Install dependencies: pip install streamlit streamlit-folium geopandas folium scipy scikit-learn pillow matplotlib branca
 3. Run: streamlit run johnsnow_dashboard_app.py
"""

import io
import os
from pathlib import Path
import base64
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from PIL import Image

import folium
from folium.plugins import HeatMap, MarkerCluster

import streamlit as st
from streamlit_folium import st_folium

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------- Paths (adjust if needed) ----------------
ROOT = Path(r"C:\Users\Admin\Downloads\MASTER\DR ERAN\Assignment JohnSnowLab")
DATA_DIR = ROOT / "Data"
OUT_DIR  = ROOT / "Outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_HTML = OUT_DIR / "final_cholera_map.html"
FINAL_PNG  = OUT_DIR / "final_cholera_map.png"

# ---------------- Helper functions ----------------
@st.cache_data(show_spinner=False)
def load_csv(fp):
    return pd.read_csv(fp)

@st.cache_data(show_spinner=False)
def load_all():
    deaths = load_csv(DATA_DIR / "deaths_by_bldg.csv")
    pumps  = load_csv(DATA_DIR / "pumps.csv")
    sewer  = load_csv(DATA_DIR / "sewergrates_ventilators.csv")
    return deaths, pumps, sewer

@st.cache_data(show_spinner=False)
def make_gdfs(deaths, pumps, sewer, xcol="COORD_X", ycol="COORD_Y", crs_proj="EPSG:27700"):
    deaths_gdf = gpd.GeoDataFrame(deaths.copy(), geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]), crs=crs_proj)
    pumps_gdf  = gpd.GeoDataFrame(pumps.copy(),  geometry=gpd.points_from_xy(pumps[xcol],  pumps[ycol]),  crs=crs_proj)
    sewer_gdf  = gpd.GeoDataFrame(sewer.copy(),  geometry=gpd.points_from_xy(sewer[xcol],  sewer[ycol]),  crs=crs_proj)
    return deaths_gdf, pumps_gdf, sewer_gdf

# Build folium map (returns folium.Map object)
def build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf,
                     kde_n=300, kde_cmap="magma", kde_opacity=0.75,
                     heat_radius=18, heat_blur=20,
                     dbscan_eps=15, dbscan_min_samples=5,
                     show_heat=True, show_kde=True, show_clusters=True, show_pumps=True, show_sewer=False):

    # KDE in projected CRS
    xs = deaths_gdf.geometry.x.values
    ys = deaths_gdf.geometry.y.values
    coords = np.vstack([xs, ys])
    kde = gaussian_kde(coords)

    xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
    xx, yy = np.mgrid[xmin:xmax:kde_n*1j, ymin:ymax:kde_n*1j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid_coords).reshape(xx.shape)

    # create temporary high-res KDE PNG and make dark pixels transparent
    fig, ax = plt.subplots(figsize=(8,8), dpi=150)
    ax.imshow(np.rot90(zz), cmap=kde_cmap, extent=[xmin, xmax, ymin, ymax], alpha=1.0)
    ax.axis('off')
    plt.tight_layout(pad=0)
    tmp_kde = OUT_DIR / "tmp_kde.png"
    fig.savefig(tmp_kde, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(tmp_kde).convert("RGBA")
    arr = np.array(img)
    threshold = 20
    mask = (arr[:,:,0] < threshold) & (arr[:,:,1] < threshold) & (arr[:,:,2] < threshold)
    arr[mask,3] = 0
    img2 = Image.fromarray(arr)
    tmp_kde_trans = OUT_DIR / "tmp_kde_trans.png"
    img2.save(tmp_kde_trans)

    # Transform bounds to WGS84
    corners = gpd.GeoSeries([Point(xmin, ymin), Point(xmax, ymax)], crs=deaths_gdf.crs).to_crs(epsg=4326)
    min_lon = float(corners.geometry.x.min()); max_lon = float(corners.geometry.x.max())
    min_lat = float(corners.geometry.y.min()); max_lat = float(corners.geometry.y.max())
    overlay_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # WGS points
    deaths_wgs = deaths_gdf.to_crs(epsg=4326)
    pumps_wgs  = pumps_gdf.to_crs(epsg=4326)
    sewer_wgs  = sewer_gdf.to_crs(epsg=4326)

    heat_data = [[row.geometry.y, row.geometry.x, 1] for _, row in deaths_wgs.iterrows()]

    center = [float(deaths_wgs.geometry.y.mean()), float(deaths_wgs.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")

    # KDE overlay
    if show_kde:
        folium.raster_layers.ImageOverlay(name="KDE raster (smooth)", image=str(tmp_kde_trans), bounds=overlay_bounds,
                                         opacity=kde_opacity, interactive=True, cross_origin=False, zindex=1).add_to(m)

    # Heatmap
    if show_heat:
        heat_layer = folium.FeatureGroup(name="HeatMap (point-weighted)", show=False)
        HeatMap(heat_data, radius=heat_radius, blur=heat_blur, max_zoom=18).add_to(heat_layer)
        heat_layer.add_to(m)

    # DBSCAN clusters
    if show_clusters:
        coords_proj = np.column_stack([deaths_gdf.geometry.x, deaths_gdf.geometry.y])
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_proj)
        deaths_gdf['cluster_tmp'] = db.labels_
        deaths_wgs['cluster_tmp'] = deaths_gdf['cluster_tmp'].values
        unique_labels = sorted([l for l in np.unique(db.labels_) if l != -1])
        cmap = matplotlib.cm.get_cmap('tab10')
        for _, row in deaths_wgs.iterrows():
            lbl = int(row.get('cluster_tmp', -1))
            if lbl != -1:
                color = mcolors.to_hex(cmap(lbl % 10)[:3])
                folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=5, color=color,
                                    fill=True, fill_opacity=0.85, weight=0.5, popup=f"Cluster: {lbl}").add_to(m)
        # centroids
        for lbl in unique_labels:
            sub = deaths_gdf[deaths_gdf['cluster_tmp'] == lbl]
            if not sub.empty:
                centroid = sub.geometry.unary_union.centroid
                centroid_wgs = gpd.GeoSeries([centroid], crs=deaths_gdf.crs).to_crs(epsg=4326).iloc[0]
                folium.Marker(location=[centroid_wgs.y, centroid_wgs.x],
                              popup=f"Cluster centroid {lbl}",
                              icon=folium.DivIcon(html=f"<div style='font-size:10px;color:black;background:rgba(255,255,255,0.85);padding:3px;border-radius:4px'>C{lbl}</div>")).add_to(m)

    # Pumps with simple svg bar popup
    if show_pumps:
        pump_group = folium.FeatureGroup(name="Pumps (click for distance chart)", show=True)
        pumps_iter = pumps_gdf.reset_index(drop=True)
        for pid, prow in pumps_iter.iterrows():
            dists = deaths_gdf.geometry.distance(prow.geometry)
            bins = [0,10,25,50,100,200,np.inf]
            labels = ["0-10","10-25","25-50","50-100","100-200",">200"]
            binned = pd.cut(dists, bins=bins, labels=labels)
            summary = binned.value_counts().reindex(labels).fillna(0).astype(int)
            bars = ''.join([f"<div style='display:inline-block;width:28px;height:{max(8,int(v*6))}px;background:#4C78A8;margin-right:2px;vertical-align:bottom' title='{labels[i]}: {v}'></div>" for i,v in enumerate(summary.values)])
            html = f"<div style='width:300px'><b>Pump {pid}</b><br>{bars}<br><small>bins: {', '.join(labels)}</small></div>"
            prow_wgs = pumps_iter.to_crs(epsg=4326).geometry.iloc[pid]
            folium.Marker(location=[prow_wgs.y, prow_wgs.x], popup=folium.Popup(html, max_width=340), icon=folium.Icon(color='darkred', icon='tint')).add_to(pump_group)
        pump_group.add_to(m)

    # Sewer
    if show_sewer:
        sewer_group = folium.FeatureGroup(name='Sewer grates', show=False)
        for _, row in sewer_gdf.to_crs(epsg=4326).iterrows():
            folium.CircleMarker([row.geometry.y, row.geometry.x], radius=3, color='green', fill=True, fill_opacity=0.6).add_to(sewer_group)
        sewer_group.add_to(m)

    # Title + Legend
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

    folium.LayerControl(collapsed=False).add_to(m)

    return m

# ---------------- Static PNG export function ----------------
def export_static_png(deaths_gdf, pumps_gdf, sewer_gdf, filename=FINAL_PNG, dpi=300):
    # create high-res static figure similar to folium style
    fig, ax = plt.subplots(figsize=(10,10), dpi=dpi)
    # KDE
    xs = deaths_gdf.geometry.x.values
    ys = deaths_gdf.geometry.y.values
    kde = gaussian_kde(np.vstack([xs, ys]))
    xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
    xx, yy = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid_coords).reshape(xx.shape)
    ax.imshow(np.rot90(zz), cmap='magma', extent=[xmin, xmax, ymin, ymax], alpha=0.6)
    # deaths
    ax.scatter(xs, ys, s=8, c='black', label='Deaths', zorder=3)
    # pumps
    ax.scatter(pumps_gdf.geometry.x, pumps_gdf.geometry.y, marker='^', s=120, c='darkred', label='Pumps', zorder=4)
    # clusters centroids
    try:
        if 'cluster_tmp' in deaths_gdf.columns:
            for lbl in sorted(set(deaths_gdf['cluster_tmp']) - {-1}):
                sub = deaths_gdf[deaths_gdf['cluster_tmp'] == lbl]
                centroid = sub.geometry.unary_union.centroid
                ax.scatter(centroid.x, centroid.y, marker='o', s=150, edgecolor='white', facecolor='none', linewidth=1.2, zorder=5)
    except Exception:
        pass
    ax.set_title('Cholera Outbreak 1854 — Density & Pump Locations', fontsize=14)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return filename

# ---------------- Streamlit App ----------------
st.set_page_config(page_title='John Snow — Cholera Dashboard', layout='wide')
st.title('John Snow 1854 — Cholera Map Dashboard')

st.sidebar.header('Controls')

# Load data
with st.spinner('Loading data...'):
    deaths, pumps, sewer = load_all()
    deaths_gdf, pumps_gdf, sewer_gdf = make_gdfs(deaths, pumps, sewer)

# Sidebar controls
show_kde = st.sidebar.checkbox('Show KDE raster', value=True)
show_heat = st.sidebar.checkbox('Show HeatMap (point-weighted)', value=False)
show_clusters = st.sidebar.checkbox('Show DBSCAN clusters', value=True)
show_pumps = st.sidebar.checkbox('Show Pumps', value=True)
show_sewer = st.sidebar.checkbox('Show Sewer grates', value=False)

kde_n = st.sidebar.slider('KDE resolution', 200, 900, 400, step=50)
kde_opacity = st.sidebar.slider('KDE opacity', 0.0, 1.0, 0.75, step=0.05)

heat_radius = st.sidebar.slider('Heat radius', 6, 40, 18, step=2)
heat_blur = st.sidebar.slider('Heat blur', 0, 40, 20, step=2)

st.sidebar.subheader('DBSCAN settings (meters)')
dbscan_eps = st.sidebar.slider('eps', 5, 100, 15, step=1)
dbscan_min = st.sidebar.slider('min_samples', 2, 10, 5, step=1)

# Buttons
if st.sidebar.button('Save current map as HTML'):
    m_save = build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf, kde_n=kde_n, kde_cmap='magma', kde_opacity=kde_opacity,
                              heat_radius=heat_radius, heat_blur=heat_blur, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min,
                              show_heat=show_heat, show_kde=show_kde, show_clusters=show_clusters, show_pumps=show_pumps, show_sewer=show_sewer)
    m_save.save(str(FINAL_HTML))
    st.success(f'Saved HTML: {FINAL_HTML}')

if st.sidebar.button('Export high-res PNG (300 dpi)'):
    with st.spinner('Exporting PNG...'):
        png_path = export_static_png(deaths_gdf, pumps_gdf, sewer_gdf, filename=FINAL_PNG, dpi=300)
    st.success(f'PNG saved: {png_path}')

# Build and render map
with st.spinner('Generating folium map...'):
    m = build_folium_map(deaths_gdf, pumps_gdf, sewer_gdf, kde_n=kde_n, kde_cmap='magma', kde_opacity=kde_opacity,
                         heat_radius=heat_radius, heat_blur=heat_blur, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min,
                         show_heat=show_heat, show_kde=show_kde, show_clusters=show_clusters, show_pumps=show_pumps, show_sewer=show_sewer)

st.subheader('Interactive Map')
st_folium(m, width='100%', height=700)

# Right column details
col1, col2 = st.columns((3,1))
with col2:
    st.subheader('Summary')
    st.markdown(f"- Total deaths: **{len(deaths_gdf)}**")
    st.markdown(f"- Total pumps: **{len(pumps_gdf)}**")
    try:
        num_clusters = int(((deaths_gdf.get('cluster_tmp', pd.Series(dtype=int)) >= 0).sum()))
        st.markdown(f"- Clusters (non-noise): **{num_clusters}**")
    except Exception:
        st.markdown(f"- Clusters: N/A")

    st.markdown('---')
    if FINAL_HTML.exists():
        with open(FINAL_HTML, 'rb') as fh:
            st.download_button('Download final_cholera_map.html', fh.read(), file_name=FINAL_HTML.name, mime='text/html')
    if FINAL_PNG.exists():
        with open(FINAL_PNG, 'rb') as fh:
            st.download_button('Download final_cholera_map.png', fh.read(), file_name=FINAL_PNG.name, mime='image/png')

