import streamlit as st
import io, base64
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from PIL import Image
import folium
from folium.plugins import HeatMap, MarkerCluster
from branca.element import Element
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="John Snow Cholera Map",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ John Snow Cholera Analysis — KDE, Clusters & Distance Bar Charts")
st.write("Version for Streamlit Cloud Deployment")

# =====================================================================
# PATH (USE RELATIVE PATH FOR STREAMLIT CLOUD)
# =====================================================================
data_dir = Path("data")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# =====================================================================
# LOAD DATA
# =====================================================================
deaths = pd.read_csv(data_dir / "deaths_by_bldg.csv")
pumps = pd.read_csv(data_dir / "pumps.csv")
sewer = pd.read_csv(data_dir / "sewergrates_ventilators.csv")

xcol = "COORD_X"
ycol = "COORD_Y"
count_col = "deaths" if "deaths" in deaths.columns else None

crs_proj = "EPSG:27700"

deaths_gdf = gpd.GeoDataFrame(deaths, geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]), crs=crs_proj)
pumps_gdf = gpd.GeoDataFrame(pumps, geometry=gpd.points_from_xy(pumps[xcol], pumps[ycol]), crs=crs_proj)
sewer_gdf = gpd.GeoDataFrame(sewer, geometry=gpd.points_from_xy(sewer[xcol], sewer[ycol]), crs=crs_proj)

# =====================================================================
# KDE
# =====================================================================
xs = deaths_gdf.geometry.x.values
ys = deaths_gdf.geometry.y.values
coords = np.vstack([xs, ys])
kde = gaussian_kde(coords)

xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
kde_n = 400

xx, yy = np.mgrid[xmin:xmax:kde_n*1j, ymin:ymax:kde_n*1j]
zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10), dpi=120)
ax.imshow(np.rot90(zz), cmap="magma", extent=[xmin, xmax, ymin, ymax], alpha=0.95)
ax.axis("off")
plt.tight_layout(pad=0)

kde_png = output_dir / "kde.png"
fig.savefig(kde_png, dpi=120, bbox_inches="tight", pad_inches=0)
plt.close(fig)

# Make transparent background
img_pil = Image.open(kde_png).convert("RGBA")
arr = np.array(img_pil)
threshold = 20
mask = (arr[:,:,0] < threshold) & (arr[:,:,1] < threshold) & (arr[:,:,2] < threshold)
arr[mask, 3] = 0
img2 = Image.fromarray(arr)
kde_png_trans = output_dir / "kde_trans.png"
img2.save(kde_png_trans)

# =====================================================================
# TRANSFORM TO WGS84 FOR FOLIUM
# =====================================================================
corners = gpd.GeoSeries([Point(xmin, ymin), Point(xmax, ymax)], crs=crs_proj).to_crs(4326)
min_lon, min_lat = corners.iloc[0].x, corners.iloc[0].y
max_lon, max_lat = corners.iloc[1].x, corners.iloc[1].y

overlay_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

deaths_wgs = deaths_gdf.to_crs(4326)
pumps_wgs = pumps_gdf.to_crs(4326)
sewer_wgs = sewer_gdf.to_crs(4326)

# =====================================================================
# HEATMAP DATA
# =====================================================================
if count_col:
    heat_data = [[row.geometry.y, row.geometry.x, float(row[count_col])] for _, row in deaths_wgs.iterrows()]
else:
    heat_data = [[row.geometry.y, row.geometry.x] for _, row in deaths_wgs.iterrows()]

# =====================================================================
# DBSCAN CLUSTERING
# =====================================================================
coords_proj = np.column_stack([deaths_gdf.geometry.x, deaths_gdf.geometry.y])
db = DBSCAN(eps=15, min_samples=5).fit(coords_proj)
deaths_gdf["cluster"] = db.labels_
deaths_wgs["cluster"] = db.labels_

# Color map for clusters
unique_labels = sorted([l for l in np.unique(db.labels_) if l != -1])
try:
    base_cmap = matplotlib.colormaps.get_cmap("tab10")
except:
    base_cmap = plt.get_cmap("tab10")

colors = [base_cmap(i/10) for i in range(10)]
cluster_color_map = {lbl: matplotlib.colors.to_hex(colors[i]) for i, lbl in enumerate(unique_labels)}

# =====================================================================
# DISTANCE BARCHART POPUP
# =====================================================================
pump_popups = {}

for pid, row in pumps_gdf.reset_index(drop=True).iterrows():
    dists = deaths_gdf.geometry.distance(row.geometry)
    bins = [0, 10, 25, 50, 100, 200, np.inf]
    labels = ["0-10","10-25","25-50","50-100","100-200",">200"]
    binned = pd.cut(dists, bins=bins, labels=labels)
    summary = binned.value_counts().reindex(labels).fillna(0)

    fig, ax = plt.subplots(figsize=(4,2.2))
    ax.bar(summary.index, summary.values)
    ax.set_title(f"Deaths near Pump {pid}", fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    pump_popups[pid] = f"<img src='data:image/png;base64,{img_b64}' width='280'/>"

# =====================================================================
# BUILD FOLIUM MAP
# =====================================================================
center = [deaths_wgs.geometry.y.mean(), deaths_wgs.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=16, tiles="CartoDB Positron")

# KDE layer
folium.raster_layers.ImageOverlay(
    image=str(kde_png_trans),
    bounds=overlay_bounds,
    opacity=0.75,
    name="KDE Density"
).add_to(m)

# Heatmap
HeatMap(heat_data, radius=15, blur=20).add_to(
    folium.FeatureGroup(name="Heatmap", show=False).add_to(m)
)

# DBSCAN clusters
cluster_layer = folium.FeatureGroup(name="Clusters").add_to(m)
for _, row in deaths_wgs.iterrows():
    lbl = row["cluster"]
    if lbl != -1:
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=cluster_color_map.get(lbl),
            fill=True
        ).add_to(cluster_layer)

# Pump markers
pump_layer = folium.FeatureGroup(name="Pumps").add_to(m)
for pid, row in pumps_wgs.reset_index(drop=True).iterrows():
    folium.Marker(
        [row.geometry.y, row.geometry.x],
        popup=folium.Popup(pump_popups[pid], max_width=350),
        icon=folium.Icon(color="red", icon="tint")
    ).add_to(pump_layer)

# Sewer
sewer_layer = folium.FeatureGroup(name="Sewer Grates", show=False).add_to(m)
for _, row in sewer_wgs.iterrows():
    folium.CircleMarker([row.geometry.y, row.geometry.x],
                        radius=3, color="green", fill=True).add_to(sewer_layer)

folium.LayerControl().add_to(m)

# =====================================================================
# EXPORT HTML
# =====================================================================
html_out = output_dir / "final_cholera_map.html"
m.save(html_out)

# =====================================================================
# DISPLAY ON STREAMLIT
# =====================================================================
st.subheader("📌 Interactive Cholera Map")

html_iframe = f"""
<iframe src="{html_out.as_posix()}" width="100%" height="700"></iframe>
"""
st.components.v1.html(html_iframe, height=750)

# =====================================================================
# DOWNLOAD BUTTON
# =====================================================================
with open(html_out, "rb") as f:
    st.download_button(
        label="⬇️ Download Full Map (HTML)",
        data=f,
        file_name="Cholera_Final_Map.html",
        mime="text/html"
    )
