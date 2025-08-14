import os
import shutil
from tempfile import NamedTemporaryFile, mkdtemp

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("ULTRALYTICS_RUNS_DIR", "/tmp/runs")

from ultralytics import settings, YOLO
import streamlit as st

valid_settings = {
    "runs_dir": os.environ.get("ULTRALYTICS_RUNS_DIR", "/tmp/runs"),
    "datasets_dir": "/tmp/datasets",
    "weights_dir": "/tmp/weights",
    "sync": False,
}
try:
    settings.update(valid_settings)
except KeyError as e:
    st.warning(f"Failed to update Ultralytics settings: {e}. Using default settings.")

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from pyproj import CRS, Transformer

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Weed Detection & Geolocation", layout="wide")

TILE_SIZE = 640             
MAX_DISPLAY = 2048            
MODEL_PATH = "best_STM.pt"   
WEED_CLASS_INDEX = 0            
BATCH_TILES = 8                 

st.markdown("""
### 📍 Weed Detection and Geolocation App
This application allows you to upload a georeferenced **orthomosaic TIFF image** (e.g., from a drone),
automatically detect **weeds** using a pre-trained **YOLO model**, and generate a shapefile with precise GPS locations of the detections.

**Key Features:**
- Upload large `.tif` orthomosaic imagery
- Detect weeds using tile-wise YOLO inference
- Convert image coordinates to GPS (WGS84)
- Download a ready-to-use **.shp file** with detections
- View an interactive visualization of weed locations and important KPIs

This tool is designed for **precision agriculture**, especially useful for **spot spraying** and **weed mapping** workflows.

Code and demo: https://github.com/rutvij-25/weedgeolocator
""")

def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def save_upload_to_temp(uploaded) -> str:
    """Stream upload to a temp file (avoid .getvalue() RAM spike)."""
    uploaded.seek(0)
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        shutil.copyfileobj(uploaded, tmp)
        return tmp.name

@st.cache_resource(show_spinner=False)
def load_model(path: str = MODEL_PATH):
    return YOLO(path)

@st.cache_data(show_spinner=False)
def run_inference_from_path(src_path: str, tile_px: int = TILE_SIZE):
    model = load_model()

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; cannot compute physical sizes.")

        cx = (src.bounds.left + src.bounds.right) / 2.0
        cy = (src.bounds.bottom + src.bounds.top) / 2.0
        to_wgs84_src = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon_c, lat_c = to_wgs84_src.transform(cx, cy)
        utm_epsg = _utm_epsg_for_lonlat(lon_c, lat_c)

        with WarpedVRT(src, dst_crs=CRS.from_epsg(utm_epsg), resampling=Resampling.nearest) as r:
            width, height = r.width, r.height
            transform = r.transform
            crs = r.crs
            
            px_w = abs(transform.a)
            px_h = abs(transform.e)

            b = r.bounds
            utm_bounds = (b.left, b.bottom, b.right, b.top)

            to_wgs84_curr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

            nx = int(np.ceil(width / tile_px))
            ny = int(np.ceil(height / tile_px))

            total_weeds = 0
            pix_x_all, pix_y_all = [], []
            utm_x_all, utm_y_all = [], []
            lons, lats = [], []

            bands_to_read = min(r.count, 3)
            batch_imgs, batch_info = [], [] 

            def flush_batch():
                nonlocal total_weeds
                if not batch_imgs:
                    return
                results_list = model.predict(
                    batch_imgs,
                    imgsz=tile_px,
                    conf=0.25,
                    iou=0.45,
                    classes=[WEED_CLASS_INDEX], 
                    verbose=False,
                )
                for res, (left_px, top_px) in zip(results_list, batch_info):
                    boxes = getattr(res, "boxes", None)
                    if boxes is None or boxes.xywh is None:
                        continue
                    xywh = np.array(boxes.xywh.tolist(), dtype=np.float32)
                    if xywh.size == 0:
                        continue

                    total_weeds += xywh.shape[0]

                    cxs = xywh[:, 0]
                    cys = xywh[:, 1]

                    full_x = left_px + cxs
                    full_y = top_px + cys
                    pix_x_all.extend(full_x.tolist())
                    pix_y_all.extend(full_y.tolist())

                    xs_arr, ys_arr = r.xy(full_y, full_x)
                    if hasattr(xs_arr, "tolist"):
                        xs_list, ys_list = xs_arr.tolist(), ys_arr.tolist()
                    else:
                        xs_list, ys_list = [xs_arr], [ys_arr]

                    utm_x_all.extend(xs_list)
                    utm_y_all.extend(ys_list)

                    lon, lat = to_wgs84_curr.transform(xs_list, ys_list)
                    if hasattr(lon, "tolist"):
                        lons.extend(lon.tolist())
                        lats.extend(lat.tolist())
                    else:
                        lons.append(lon)
                        lats.append(lat)

                batch_imgs.clear()
                batch_info.clear()

            for iy in range(ny):
                for ix in range(nx):
                    left_px = ix * tile_px
                    top_px = iy * tile_px
                    right_px = min(left_px + tile_px, width)
                    bottom_px = min(top_px + tile_px, height)

                    win_w = right_px - left_px
                    win_h = bottom_px - top_px
                    if win_w <= 0 or win_h <= 0:
                        continue

                    mask = r.read_masks(
                        1,
                        window=Window(left_px, top_px, win_w, win_h),
                        out_shape=(win_h, win_w),
                        resampling=Resampling.nearest,
                    )
                    if (mask == 0).mean() > 0.95:
                        continue

                    chip = r.read(
                        indexes=list(range(1, bands_to_read + 1)),
                        window=Window(left_px, top_px, win_w, win_h),
                        out_shape=(bands_to_read, win_h, win_w),
                        resampling=Resampling.nearest,
                    )
                    chip = np.transpose(chip, (1, 2, 0)).copy() 

                    if chip.shape[2] == 1:
                        chip = np.repeat(chip, 3, axis=2)
                    elif chip.shape[2] > 3:
                        chip = chip[:, :, :3]

                    chip = np.ascontiguousarray(chip)
                    if chip.dtype != np.uint8:
                        chip = np.clip(chip, 0, 255).astype(np.uint8)

                    batch_imgs.append(chip)
                    batch_info.append((left_px, top_px))

                    if len(batch_imgs) >= BATCH_TILES:
                        flush_batch()

            flush_batch()

            field_area_m2 = (width * px_w) * (height * px_h)
            avg_density = (total_weeds / field_area_m2) if field_area_m2 > 0 else 0.0

            meta = {
                "width": width,
                "height": height,
                "px_w": px_w,
                "px_h": px_h,
                "utm_epsg": utm_epsg,
                "utm_bounds": utm_bounds,  
                "tile_size_px": tile_px,
                "transform": tuple(transform),
            }

    return {
        "total_weeds": int(total_weeds),
        "field_area_m2": float(field_area_m2),
        "avg_density": float(avg_density),
        "lons": lons,
        "lats": lats,
        "pix_x": pix_x_all,
        "pix_y": pix_y_all,
        "utm_x": utm_x_all,
        "utm_y": utm_y_all,
        "meta": meta,
    }

@st.cache_data(show_spinner=False)
def compute_thumbnail_from_path(src_path: str, max_display: int, utm_epsg: int):
    """Produce a small RGB thumbnail in the same UTM CRS as inference."""
    with rasterio.open(src_path) as src:
        with WarpedVRT(src, dst_crs=CRS.from_epsg(utm_epsg), resampling=Resampling.nearest) as ds:
            scale = min(max_display / ds.width, max_display / ds.height, 1.0)
            out_w = max(1, int(ds.width * scale))
            out_h = max(1, int(ds.height * scale))

            bands_to_read = min(ds.count, 3)
            thumb = ds.read(
                indexes=list(range(1, bands_to_read + 1)),
                out_shape=(bands_to_read, out_h, out_w),
                resampling=Resampling.nearest,
            )
            thumb = np.transpose(thumb, (1, 2, 0))
            if thumb.shape[2] == 1:
                thumb = np.repeat(thumb, 3, axis=2)

            scale_x = out_w / ds.width
            scale_y = out_h / ds.height

    return thumb, (out_w, out_h), (scale_x, scale_y)

@st.cache_data(show_spinner=False)
def build_counts_safe(utm_x, utm_y, utm_bounds, max_cells=60_000_000):
    """
    Build a counts grid in UTM meters with dynamic coarsening to keep memory safe.
    Returns (counts_grid, x0, y0, cell_m) where each cell is cell_m × cell_m meters.
    """
    left, bottom, right, top = utm_bounds
    x0 = np.floor(left)
    y0 = np.floor(bottom)

    nx = int(np.ceil(right) - x0)
    ny = int(np.ceil(top) - y0)
    nx = max(nx, 1)
    ny = max(ny, 1)

    cell_m = 1
    if nx * ny > max_cells:
        cell_m = int(np.ceil(np.sqrt((nx * ny) / max_cells)))

    nx_c = max(1, int(np.ceil(nx / cell_m)))
    ny_c = max(1, int(np.ceil(ny / cell_m)))
    counts = np.zeros((ny_c, nx_c), dtype=np.uint32)

    if len(utm_x) == 0:
        return counts, x0, y0, cell_m

    utm_x = np.asarray(utm_x, dtype=np.float64)
    utm_y = np.asarray(utm_y, dtype=np.float64)

    ii = np.floor((utm_x - x0) / cell_m).astype(np.int64)
    jj = np.floor((utm_y - y0) / cell_m).astype(np.int64)
    ii = np.clip(ii, 0, nx_c - 1)
    jj = np.clip(jj, 0, ny_c - 1)

    np.add.at(counts, (jj, ii), 1)
    return counts, x0, y0, cell_m

def aggregate_grid_sum(grid: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return grid
    ny, nx = grid.shape
    pad_y = (-ny) % k
    pad_x = (-nx) % k
    if pad_y or pad_x:
        grid = np.pad(grid, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
    ny_p, nx_p = grid.shape
    return grid.reshape(ny_p // k, k, nx_p // k, k).sum(axis=(1, 3))

@st.cache_data(show_spinner=True)
def build_shapefile_zip(lons, lats):
    gdf = gpd.GeoDataFrame(
        {"id": list(range(1, len(lons) + 1)), "lat": lats, "lon": lons},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    tmp_dir = mkdtemp()
    shp_base = os.path.join(tmp_dir, "detections")
    out_path = shp_base + ".zip"
    try:
        gdf.to_file(filename=shp_base, driver="ESRI Shapefile", engine="pyogrio")
        shutil.make_archive(shp_base, "zip", tmp_dir, "detections")
    except Exception:
        geojson_path = shp_base + ".geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        shutil.make_archive(shp_base, "zip", tmp_dir, "detections.geojson")
    with open(out_path, "rb") as f:
        shp_zip_bytes = f.read()
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    return shp_zip_bytes

uploaded = st.file_uploader("Upload Orthomosaic GeoTIFF", type=["tif", "tiff"])
if uploaded is None:
    st.info("Upload a georeferenced TIFF to begin.")
    st.stop()

with st.spinner("Processing… (tiling, YOLO inference, metrics, preview)"):
    src_path = save_upload_to_temp(uploaded)
    try:
        res = run_inference_from_path(src_path, TILE_SIZE)
        total_weeds = res["total_weeds"]
        field_area_m2 = res["field_area_m2"]
        avg_density = res["avg_density"]
        lons, lats = res["lons"], res["lats"]
        pix_x = np.array(res["pix_x"])
        pix_y = np.array(res["pix_y"])
        utm_x = res["utm_x"]
        utm_y = res["utm_y"]
        meta = res["meta"]

        width = meta["width"]
        height = meta["height"]
        px_w = meta["px_w"]
        px_h = meta["px_h"]
        utm_epsg = meta["utm_epsg"]
        utm_bounds = meta["utm_bounds"]  

        thumb, (disp_w, disp_h), (scale_x, scale_y) = compute_thumbnail_from_path(
            src_path=src_path,
            max_display=MAX_DISPLAY,
            utm_epsg=utm_epsg
        )

        shp_zip_bytes = build_shapefile_zip(lons, lats)

        counts_base, x0_m, y0_m, cell_m = build_counts_safe(utm_x, utm_y, utm_bounds)

    finally:
        try:
            os.unlink(src_path)
        except Exception:
            pass

st.success("Done!")

st.markdown("#### Download detections")
st.download_button(
    label="⬇️ Download detections_shapefile.zip",
    data=shp_zip_bytes,
    file_name="detections_shapefile.zip",
    mime="application/zip",
)
csv_bytes = pd.DataFrame({"id": range(1, len(lons)+1), "lat": lats, "lon": lons}).to_csv(index=False).encode()
st.download_button(
    label="⬇️ Download detections.csv",
    data=csv_bytes,
    file_name="detections.csv",
    mime="text/csv",
)

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Total weeds detected", f"{total_weeds:,}")
c2.metric("Field size (m²)", f"{field_area_m2:,.0f}")
c3.metric("Avg density (weeds/m²)", f"{avg_density:.6f}")

st.subheader("Orthomosaic + Detections (points)")

fig_points = px.imshow(thumb)
fig_points.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
fig_points.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)

if pix_x.size > 0:
    sx = pix_x * scale_x
    sy = pix_y * scale_y
    fig_points.add_trace(
        go.Scatter(
            x=sx, y=sy, mode="markers", name="Weeds",
            marker=dict(size=3),
            hoverinfo="skip",
        )
    )
st.plotly_chart(fig_points, use_container_width=True)

st.subheader("Orthomosaic + Weed Density Heatmap")

st.caption(f"Base grid cell = **{cell_m} m**. Use the slider to aggregate by multiples of this base size.")
overlay_cells = st.slider(
    "Overlay aggregation (× base cell)",
    min_value=1, max_value=20, value=1, step=1,
    help="Visualization aggregation factor. Histogram stays at base cell size."
)
alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05, key="opacity_slider")

overlay_grid = aggregate_grid_sum(counts_base, overlay_cells)
overlay_cell_m = overlay_cells * cell_m

colorbar_title = "weed count (per cell)"
hover_tmpl = f"{overlay_cell_m}×{overlay_cell_m} m cell<br>count: %{{z}}<extra></extra>"
ny_o, nx_o = overlay_grid.shape
x_centers_m = x0_m + (overlay_cell_m * (np.arange(nx_o) + 0.5))
y_centers_m = y0_m + (overlay_cell_m * (np.arange(ny_o) + 0.5))

left_m, bottom_m, right_m, top_m = utm_bounds
x_cols = (x_centers_m - left_m) / px_w
y_rows = (top_m - y_centers_m) / px_h
x_disp = x_cols * scale_x
y_disp = y_rows * scale_y

fig_overlay = px.imshow(thumb)
fig_overlay.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
fig_overlay.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)

fig_overlay.add_trace(
    go.Heatmap(
        z=overlay_grid,   
        x=x_disp,          
        y=y_disp,
        colorbar=dict(title=colorbar_title),
        opacity=alpha,
        reversescale=True,
        hovertemplate=hover_tmpl,
        zmin=0
    )
)
fig_overlay.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False)
st.plotly_chart(fig_overlay, use_container_width=True)

st.subheader(f"Histogram • Weeds per {cell_m} m² tile")
tile_counts_flat = counts_base.flatten()
freq = np.bincount(tile_counts_flat) 
k_vals = np.arange(freq.size)
fig_1m_hist = px.bar(
    x=k_vals, y=freq,
    labels={"x": f"Weeds per {cell_m} m² tile", "y": "Number of tiles"},
)
fig_1m_hist.update_layout(bargap=0.05)
st.plotly_chart(fig_1m_hist, use_container_width=True)
