# frontend/app.py
import os, re, unicodedata, urllib.parse, io, base64, tempfile, pathlib, shutil
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import requests
import httpx
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import streamlit as st

st.set_page_config(page_title="Weed Detection & Geolocation", layout="wide")

SHOW_DEBUG = bool(int(os.getenv("UI_DEBUG", "0")))   

def _clean(s: str) -> str:
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")  
    return s.strip().rstrip("/")

def resolve_api_base() -> str:
    candidate = os.getenv("API_BASE") or "http://127.0.0.1:8000"
    candidate = _clean(candidate)
    if not re.match(r"^https?://", candidate):
        candidate = "http://" + candidate
    candidate = candidate.replace("0.0.0.0", "127.0.0.1") 
    parsed = urllib.parse.urlparse(candidate)
    if not parsed.scheme or not parsed.netloc or " " in candidate:
        st.error("Backend URL is invalid. Set API_BASE like http://127.0.0.1:8000")
        st.stop()
    return candidate

API_BASE = resolve_api_base()


requests_sess = requests.Session()
requests_sess.trust_env = False  

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=15.0, read=600.0, write=600.0, pool=15.0),
    follow_redirects=False,
    transport=httpx.HTTPTransport(retries=0),
    trust_env=False,
    http2=False,
)


try:
    hr = requests_sess.get(
        f"{API_BASE}/health",
        timeout=(5, 5),
        allow_redirects=False,
        proxies={"http": None, "https": None},
        headers={"Connection": "close"},
    )
    hr.raise_for_status()
except Exception:
    st.error("Cannot reach the backend service. Make sure the API is running at API_BASE.")
    st.stop()


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

def sanitize_filename(name: str | None) -> str:
    if not name:
        return "image.tif"
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name) or "image.tif"

def save_uploaded_to_temp(uploaded_file) -> tuple[str, str]:

    safe_name = sanitize_filename(getattr(uploaded_file, "name", None))
    suffix = pathlib.Path(safe_name).suffix or ".tif"
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # copy in chunks to avoid reading whole file into memory
        shutil.copyfileobj(uploaded_file, tmp, length=16 * 1024 * 1024)  # 16MB chunks
        return tmp.name, safe_name

def post_file_resilient(infer_url: str, temp_path: str, safe_name: str) -> dict:
   
    last_exc = None

    try:
        with open(temp_path, "rb") as fh:
            files = {"file": (safe_name, fh, "image/tiff")}
            r = requests_sess.post(
                infer_url,
                files=files,
                timeout=(15, 600),
                allow_redirects=False,
                proxies={"http": None, "https": None},
                headers={"Connection": "close"},
            )
        r.raise_for_status()
        return r.json()
    except Exception as e1:
        last_exc = e1
        if SHOW_DEBUG: st.info(f"Strategy #1 failed: {repr(e1)}")

   
    try:
        with open(temp_path, "rb") as fh:
            encoder = MultipartEncoder(fields={"file": (safe_name, fh, "image/tiff")})
            monitor = MultipartEncoderMonitor(encoder, lambda m: None)
            headers = {"Content-Type": monitor.content_type, "Connection": "close"}
            r = requests_sess.post(
                infer_url,
                data=monitor,
                timeout=(15, 600),
                allow_redirects=False,
                proxies={"http": None, "https": None},
                headers=headers,
            )
        r.raise_for_status()
        return r.json()
    except Exception as e2:
        last_exc = e2
        if SHOW_DEBUG: st.info(f"Strategy #2 failed: {repr(e2)}")

    try:
        with open(temp_path, "rb") as fh:
            files = {"file": (safe_name, fh, "image/tiff")}
            r = httpx_client.post(infer_url, files=files, headers={"Connection": "close"})
        r.raise_for_status()
        return r.json()
    except Exception as e3:
        last_exc = e3
        if SHOW_DEBUG: st.info(f"Strategy #3 failed: {repr(e3)}")
    

    raise RuntimeError("Upload failed. Please verify the backend is reachable and try again.") from last_exc

# ---------- UI ----------
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

uploaded = st.file_uploader("Upload Orthomosaic GeoTIFF", type=["tif", "tiff"])
if uploaded is None:
    st.info("Upload a georeferenced TIFF to begin.")
    st.stop()

with st.spinner("Processing…"):
    temp_path, safe_name = save_uploaded_to_temp(uploaded)
    try:
        data = post_file_resilient(f"{API_BASE}/infer", temp_path, safe_name)
    except Exception as e:
        
        st.error(str(e))
        if SHOW_DEBUG:
            st.exception(e)
        
        try: os.unlink(temp_path)
        except Exception: pass
        st.stop()
    finally:
        try: os.unlink(temp_path)
        except Exception: pass

thumb = Image.open(io.BytesIO(base64.b64decode(data["thumb_png_b64"])))
thumb_np = np.array(thumb)
shp_zip_bytes = base64.b64decode(data["shp_zip_b64"])


disp_w, disp_h = data["disp_w"], data["disp_h"]
scale_x, scale_y = data["scale_x"], data["scale_y"]
pix_x = np.array(data["pix_x"], dtype=np.float64)
pix_y = np.array(data["pix_y"], dtype=np.float64)
counts_base = np.array(data["counts_base"], dtype=np.int64)
x0_m, y0_m = data["x0_m"], data["y0_m"]
cell_m = data["cell_m"]
left_m, bottom_m, right_m, top_m = data["meta"]["utm_bounds"]
px_w, px_h = data["meta"]["px_w"], data["meta"]["px_h"]

st.success("Completed.")

st.markdown("#### Download detections")
st.download_button(
    label="Download detections_shapefile.zip",
    data=shp_zip_bytes,
    file_name="detections_shapefile.zip",
    mime="application/zip",
)


st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Total weeds detected", f"{data['total_weeds']:,}")
c2.metric("Field size (m²)", f"{data['field_area_m2']:,.0f}")
c3.metric("Avg density (weeds/m²)", f"{data['avg_density']:.6f}")

st.subheader("Orthomosaic + Detections (points)")
fig_points = px.imshow(thumb_np)
fig_points.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
fig_points.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
if pix_x.size > 0:
    sx = pix_x * scale_x
    sy = pix_y * scale_y
    fig_points.add_trace(
        go.Scatter(x=sx, y=sy, mode="markers", name="Weeds", marker=dict(size=3), hoverinfo="skip")
    )
st.plotly_chart(fig_points, use_container_width=True)

st.subheader("Orthomosaic + Weed Density Heatmap")
st.caption(f"Base grid cell = **{cell_m} m**. Use the slider to aggregate by multiples of this base size.")
overlay_cells = st.slider("Overlay aggregation (× base cell)", 1, 20, 1, 1)
alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05, key="opacity_slider")

overlay_grid = aggregate_grid_sum(counts_base, overlay_cells)
overlay_cell_m = overlay_cells * cell_m

ny_o, nx_o = overlay_grid.shape
x_centers_m = x0_m + (overlay_cell_m * (np.arange(nx_o) + 0.5))
y_centers_m = y0_m + (overlay_cell_m * (np.arange(ny_o) + 0.5))


x_disp = (x_centers_m - left_m) / px_w * scale_x
y_disp = (top_m - y_centers_m) / px_h * scale_y

fig_overlay = px.imshow(thumb_np)
fig_overlay.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
fig_overlay.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
fig_overlay.add_trace(
    go.Heatmap(
        z=overlay_grid, x=x_disp, y=y_disp,
        colorscale="Blues", reversescale=False,
        zmin=0, zmax=float(np.nanmax(overlay_grid)) if overlay_grid.size else 1.0,
        colorbar=dict(title="weed count (per cell)"),
        opacity=alpha,
        hovertemplate=f"{overlay_cell_m}×{overlay_cell_m} m cell<br>count: %{{z}}<extra></extra>"
    )
)
fig_overlay.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False)
st.plotly_chart(fig_overlay, use_container_width=True)

st.subheader(f"Histogram • Weeds per {cell_m} m² tile")
tile_counts_flat = counts_base.flatten()
freq = np.bincount(tile_counts_flat)
k_vals = np.arange(freq.size)
fig_1m_hist = px.bar(x=k_vals, y=freq, labels={"x": f"Weeds per {cell_m} m² tile", "y": "Number of tiles"})
fig_1m_hist.update_layout(bargap=0.05)
st.plotly_chart(fig_1m_hist, use_container_width=True)
