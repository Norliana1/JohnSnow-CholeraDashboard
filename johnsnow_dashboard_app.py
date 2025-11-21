import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Matplotlib-specific deprecation warnings
try:
    import matplotlib
    from matplotlib import MatplotlibDeprecationWarning
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
except Exception:
    pass

import os
from pathlib import Path
import io, base64
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap, MarkerCluster
from PIL import Image
import matplotlib.colors as mcolors

# ---------------- Paths ----------------
data_dir = Path(r"C:\Users\Admin\Downloads\MASTER\DR ERAN\Assignment JohnSnowLab\Data")
out_dir  = Path(r"C:\Users\Admin\Outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------- Load CSVs ----------------
deaths = pd.read_csv(data_dir / "deaths_by_bldg.csv")
pumps  = pd.read_csv(data_dir / "pumps.csv")
sewer  = pd.read_csv(data_dir / "sewergrates_ventilators.csv")

# Columns (adjust if your column names differ)
xcol = "COORD_X"
ycol = "COORD_Y"
count_col = "deaths" if "deaths" in deaths.columns else None

# ---------------- Create GeoDataFrames (projected CRS: EPSG:27700) ----------------
crs_proj = "EPSG:27700"
deaths_gdf = gpd.GeoDataFrame(deaths.copy(),
                              geometry=gpd.points_from_xy(deaths[xcol], deaths[ycol]),
                              crs=crs_proj)
pumps_gdf  = gpd.GeoDataFrame(pumps.copy(),
                              geometry=gpd.points_from_xy(pumps[xcol], pumps[ycol]),
                              crs=crs_proj)
sewer_gdf  = gpd.GeoDataFrame(sewer.copy(),
                              geometry=gpd.points_from_xy(sewer[xcol], sewer[ycol]),
                              crs=crs_proj)

# ---------------- KDE (projected CRS) - resolution (tune if slow) ----------------
xs = deaths_gdf.geometry.x.values
ys = deaths_gdf.geometry.y.values
coords = np.vstack([xs, ys])
kde = gaussian_kde(coords)

xmin, ymin, xmax, ymax = deaths_gdf.total_bounds
# WARNING: larger kde_n -> higher resolution and memory/time cost
kde_n = 400   # set to 700 if your machine can handle it; otherwise 300-400 recommended
xx, yy = np.mgrid[xmin:xmax:kde_n*1j, ymin:ymax:kde_n*1j]
grid_coords = np.vstack([xx.ravel(), yy.ravel()])
zz = kde(grid_coords).reshape(xx.shape)

# Save KDE raster (hot colormap) and then make dark pixels transparent
fig, ax = plt.subplots(figsize=(10,10), dpi=150)
ax.imshow(np.rot90(zz), cmap="magma", extent=[xmin, xmax, ymin, ymax], alpha=0.95)
ax.axis('off')
plt.tight_layout(pad=0)
kde_png = out_dir / "kde_raster_polished.png"
fig.savefig(kde_png, dpi=150, bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Make dark pixels transparent to allow basemap/points to show through
img_pil = Image.open(kde_png).convert("RGBA")
arr = np.array(img_pil)

# If the array is grayscale, convert it to RGBA (adding alpha channel)
if len(arr.shape) == 2:  # grayscale image
    arr = np.stack([arr] * 3, axis=-1)  # Convert grayscale to RGB by repeating the single channel
    arr = np.concatenate([arr, np.full_like(arr[:, :, :1], 255)], axis=-1)  # Add an alpha channel (fully opaque)
elif arr.shape[2] == 3:  # RGB image
    arr = np.concatenate([arr, np.full_like(arr[:, :, :1], 255)], axis=-1)  # Add an alpha channel (fully opaque)

# Save the new transparent image
img_pil2 = Image.fromarray(arr)
kde_png_trans = out_dir / "kde_raster_transparent_polished.png"
img_pil2.save(kde_png_trans)

print("Saved KDE raster (transparent):", kde_png_trans)

# ---------------- Transform bounds to EPSG:4326 for ImageOverlay ----------------
corners = gpd.GeoSeries([Point(xmin, ymin), Point(xmax, ymax)], crs=crs_proj)
corners_wgs = corners.to_crs(epsg=4326)
min_lon = float(corners_wgs.geometry.x.min())
max_lon = float(corners_wgs.geometry.x.max())
min_lat = float(corners_wgs.geometry.y.min())
max_lat = float(corners_wgs.geometry.y.max())
overlay_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

# ---------------- WGS84 points for folium layers ----------------
deaths_wgs = deaths_gdf.to_crs(epsg=4326)
pumps_wgs  = pumps_gdf.to_crs(epsg=4326)
sewer_wgs  = sewer_gdf.to_crs(epsg=4326)

# Heat data (weighted if death count column exists)
if count_col:
    heat_data = [[row.geometry.y, row.geometry.x, float(row[count_col])] for _, row in deaths_wgs.iterrows()]
else:
    heat_data = [[row.geometry.y, row.geometry.x, 1] for _, row in deaths_wgs.iterrows()]

# ---------------- DBSCAN clustering (projected CRS) ----------------
coords_proj = np.column_stack([deaths_gdf.geometry.x, deaths_gdf.geometry.y])
db = DBSCAN(eps=15, min_samples=5).fit(coords_proj)   # eps in meters
deaths_gdf['cluster'] = db.labels_
deaths_wgs['cluster'] = deaths_gdf['cluster'].values

# ---------------- Prepare color map for clusters (compatible method) ----------------
unique_labels = sorted([l for l in np.unique(db.labels_) if l != -1])
# get base colormap (compatible across matplotlib versions)
try:
    base_cmap = matplotlib.colormaps.get_cmap("tab10")
except Exception:
    base_cmap = plt.get_cmap("tab10")
num_colors = max(3, len(unique_labels))
colors = [base_cmap(i / num_colors) for i in range(num_colors)]
cluster_color_map = {}
for i, label in enumerate(unique_labels):
    rgb = colors[i][:3]
    cluster_color_map[label] = mcolors.to_hex(rgb)

# ---------------- Precompute distance bins per pump (popup chart) ----------------
pump_popups = {}
pumps_iter = pumps_gdf.reset_index(drop=True)
for pid, prow in pumps_iter.iterrows():
    dists = deaths_gdf.geometry.distance(prow.geometry)  # meters
    bins = [0, 10, 25, 50, 100, 200, np.inf]
    labels = ["0-10","10-25","25-50","50-100","100-200",">200"]
    binned = pd.cut(dists, bins=bins, labels=labels)
    summary = binned.value_counts().reindex(labels).fillna(0).astype(int)
    # create compact bar chart image
    fig, ax = plt.subplots(figsize=(4,2.2))
    ax.bar(summary.index.astype(str), summary.values, color="#4C78A8")
    ax.set_title(f"Deaths by distance to Pump {pid}", fontsize=9)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    html = f'<div style="width:320px">{img_b64 and "<img src=\'data:image/png;base64," + img_b64 + "\' style=\'width:100%\'/>"}</div>'
    pump_popups[pid] = html

# ---------------- Build Folium map ----------------
center = [float(deaths_wgs.geometry.y.mean()), float(deaths_wgs.geometry.x.mean())]
m = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")

# 1) Add KDE raster overlay (transparent PNG)
folium.raster_layers.ImageOverlay(
    name="KDE raster (smooth)",
    image=str(kde_png_trans),
    bounds=overlay_bounds,
    opacity=0.75,
    interactive=True,
    cross_origin=False,
    zindex=1
).add_to(m)

# 2) HeatMap (point-weighted) layer (off by default)
heat_layer = folium.FeatureGroup(name="HeatMap (point-weighted)", show=False)
HeatMap(heat_data, radius=18, blur=20, max_zoom=18).add_to(heat_layer)
heat_layer.add_to(m)

# 3) DBSCAN cluster layer (points + centroid markers)
cluster_layer = folium.FeatureGroup(name="DBSCAN clusters", show=True)
for _, row in deaths_wgs.iterrows():
    lbl = int(row.get("cluster", -1))
    if lbl != -1:
        color = cluster_color_map.get(lbl, "#3186cc")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.85,
            weight=0.5,
            popup=f"Cluster: {lbl}"
        ).add_to(cluster_layer)

# centroids using safe union method (union_all if available)
if len(unique_labels) > 0:
    for lbl in unique_labels:
        sub = deaths_gdf[deaths_gdf['cluster'] == lbl]
        if not sub.empty:
            try:
                # prefer union_all (newer geopandas)
                if hasattr(sub.geometry, "union_all"):
                    centroid = sub.geometry.union_all().centroid
                else:
                    centroid = sub.geometry.unary_union.centroid
            except Exception:
                centroid = sub.geometry.unary_union.centroid
            # centroid might be MultiPoint/Point - get first geometry
            if hasattr(centroid, "geoms"):
                centroid = list(centroid.geoms)[0]
            centroid_wgs = gpd.GeoSeries([centroid], crs=crs_proj).to_crs(epsg=4326).iloc[0]
            folium.Marker(
                location=[centroid_wgs.y, centroid_wgs.x],
                popup=f"Cluster centroid {lbl}",
                icon=folium.DivIcon(html=f"""<div style="font-size:10px;color:black;background:rgba(255,255,255,0.85);padding:3px;border-radius:4px">C{lbl}</div>""")
            ).add_to(cluster_layer)

cluster_layer.add_to(m)

# 4) Pump markers + popup (with bar chart image)
pump_layer = folium.FeatureGroup(name="Pumps (click for distance chart)", show=True)
pumps_wgs_reset = pumps_gdf.reset_index(drop=True).to_crs(epsg=4326)
for pid, prow in pumps_wgs_reset.iterrows():
    popup_html = pump_popups.get(pid, "<div>No data</div>")
    folium.Marker(
        location=[prow.geometry.y, prow.geometry.x],
        popup=folium.Popup(popup_html, max_width=360),
        icon=folium.Icon(color="darkred", icon="tint")
    ).add_to(pump_layer)
pump_layer.add_to(m)

# 5) All death points via MarkerCluster for exploration (small black dots)
mc = MarkerCluster(name="All death points (clustered)").add_to(m)
for _, row in deaths_wgs.iterrows():
    popup = ""
    if "ID" in row:
        popup += f"ID: {row['ID']}<br>"
    if count_col:
        popup += f"Deaths: {row[count_col]}<br>"
    popup += f"Cluster: {row.get('cluster', '')}"
    folium.CircleMarker([row.geometry.y, row.geometry.x],
                        radius=3, color="black", fill=True, fill_opacity=0.65,
                        popup=popup).add_to(mc)

# 6) Sewer layer (optional)
sewer_layer = folium.FeatureGroup(name="Sewer grates", show=False)
for _, row in sewer_wgs.iterrows():
    folium.CircleMarker([row.geometry.y, row.geometry.x],
                        radius=3, color="green", fill=True, fill_opacity=0.6).add_to(sewer_layer)
sewer_layer.add_to(m)

# ---------------- Custom Legend (HTML) ----------------
legend_html = """
<div style="
    position: fixed;
    bottom: 50px;
    left: 10px;
    width: 240px;
    z-index:9999;
    font-size:12px;
    ">
    <div style="background:white; padding:10px; border-radius:8px; box-shadow:2px 2px 8px rgba(0,0,0,0.25)">
      <b>Legend</b><br><br>
      <span style="display:inline-block;width:14px;height:14px;background:rgba(180,30,20,0.75);margin-right:8px;border-radius:3px"></span> KDE raster hotspot<br>
      <span style="display:inline-block;width:10px;height:10px;background:#000;margin-right:8px;border-radius:2px"></span> Death points<br>
      <svg width="14" height="14" style="vertical-align:middle;margin-right:6px;"><polygon points="7,0 14,14 0,14" style="fill:darkred"/></svg> Pumps<br>
      <span style="display:inline-block;width:10px;height:10px;background:green;margin-right:8px;border-radius:2px"></span> Sewer grates<br>
      <div style="margin-top:6px; font-size:11px; color:#555;">Toggle layers using top-right control.</div>
    </div>
</div>
"""
from branca.element import Element

title_html = """
     <h3 align="center" style="font-size:20px">
         <b>Cholera Outbreak Analysis — Density, Clusters & Distance Patterns</b>
     </h3>
     """
m.get_root().html.add_child(Element(title_html))

m.get_root().html.add_child(folium.Element(legend_html))

# Layer control and save
folium.LayerControl(collapsed=False).add_to(m)
html_out = out_dir / "final_cholera_map.html"
m.save(str(html_out))
print("Saved:", html_out)

m
