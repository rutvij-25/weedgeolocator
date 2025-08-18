import os
import io
import base64
import shutil
from tempfile import NamedTemporaryFile, mkdtemp

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("ULTRALYTICS_RUNS_DIR", "/tmp/runs")

from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import settings, YOLO

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from pyproj import CRS, Transformer
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = BACKEND_DIR.parent                     

MODEL_PATH = PROJECT_ROOT / "models" / "best_STM.pt"

TILE_SIZE = int(os.getenv("TILE_SIZE", "640"))
WEED_CLASS_INDEX = int(os.getenv("WEED_CLASS_INDEX", "0"))
BATCH_TILES = int(os.getenv("BATCH_TILES", "8"))
MAX_DISPLAY = int(os.getenv("MAX_DISPLAY", "2048"))
MAX_CELLS = int(os.getenv("MAX_CELLS", "60000000"))

valid_settings = {
    "runs_dir": os.environ.get("ULTRALYTICS_RUNS_DIR", "/tmp/runs"),
    "datasets_dir": "/tmp/datasets",
    "weights_dir": "/tmp/weights",
    "sync": False,
}
try:
    settings.update(valid_settings)
except KeyError:
    pass

app = FastAPI(title="Weed Geolocator API", version="1.0.0")

origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def save_upload_to_temp(uploaded: UploadFile) -> str:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        uploaded.file.seek(0)
        shutil.copyfileobj(uploaded.file, tmp)
        return tmp.name

_MODEL = None
def load_model(path: str = MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(path)
    return _MODEL

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
                    cxs = xywh[:, 0]; cys = xywh[:, 1]

                    
                    full_x = left_px + cxs
                    full_y = top_px + cys
                    pix_x_all.extend(full_x.tolist())
                    pix_y_all.extend(full_y.tolist())

                    
                    xs_arr, ys_arr = r.xy(full_y, full_x)
                    xs_arr = np.atleast_1d(xs_arr).astype(np.float64).ravel()
                    ys_arr = np.atleast_1d(ys_arr).astype(np.float64).ravel()
                    utm_x_all.extend(xs_arr.tolist())
                    utm_y_all.extend(ys_arr.tolist())

                   
                    lon, lat = to_wgs84_curr.transform(xs_arr, ys_arr)
                    lon = np.atleast_1d(lon).astype(np.float64).ravel()
                    lat = np.atleast_1d(lat).astype(np.float64).ravel()
                    ok = np.isfinite(lon) & np.isfinite(lat)
                    if np.any(ok):
                        lons.extend(lon[ok].tolist())
                        lats.extend(lat[ok].tolist())

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
                "width": int(width),
                "height": int(height),
                "px_w": float(px_w),
                "px_h": float(px_h),
                "utm_epsg": int(utm_epsg),
                "utm_bounds": tuple(float(x) for x in utm_bounds),
                "tile_size_px": int(tile_px),
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

def compute_thumbnail_from_path(src_path: str, max_display: int, utm_epsg: int):
    """Return PNG bytes of a small RGB thumbnail in UTM CRS, plus dims/scale."""
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

            img = Image.fromarray(np.clip(thumb, 0, 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            scale_x = out_w / ds.width
            scale_y = out_h / ds.height

    return png_bytes, (out_w, out_h), (scale_x, scale_y)

def build_counts_safe(utm_x, utm_y, utm_bounds, max_cells=MAX_CELLS):
    left, bottom, right, top = utm_bounds
    x0 = float(np.floor(left))
    y0 = float(np.floor(bottom))

    nx = int(np.ceil(right) - x0)
    ny = int(np.ceil(top) - y0)
    nx = max(nx, 1); ny = max(ny, 1)

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

def build_shapefile_zip(lons, lats) -> bytes:
    """Create a zipped Shapefile or fallback GeoJSON; robust to nesting."""
    def _flatten_to_float1d(seq):
        flat = []
        for item in seq:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend(np.asarray(item, dtype=np.float64).ravel().tolist())
            else:
                flat.append(float(item))
        return np.asarray(flat, dtype=np.float64)

    lon_arr = _flatten_to_float1d(lons)
    lat_arr = _flatten_to_float1d(lats)
    n = min(lon_arr.size, lat_arr.size)
    lon_arr, lat_arr = lon_arr[:n], lat_arr[:n]
    finite = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    lon_arr, lat_arr = lon_arr[finite], lat_arr[finite]

    tmp_dir = mkdtemp()
    shp_base = os.path.join(tmp_dir, "detections")
    out_zip_path = shp_base + ".zip"

    try:
        if lon_arr.size == 0:
            empty_gdf = gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")
            geojson_path = shp_base + ".geojson"
            empty_gdf.to_file(geojson_path, driver="GeoJSON")
            shutil.make_archive(shp_base, "zip", tmp_dir, "detections.geojson")
        else:
            gdf = gpd.GeoDataFrame(
                {"id": np.arange(1, lon_arr.size + 1), "lat": lat_arr, "lon": lon_arr},
                geometry=gpd.points_from_xy(lon_arr, lat_arr),
                crs="EPSG:4326",
            )
            try:
                gdf.to_file(filename=shp_base, driver="ESRI Shapefile", engine="pyogrio")
                shutil.make_archive(shp_base, "zip", tmp_dir, "detections")
            except Exception:
                geojson_path = shp_base + ".geojson"
                gdf.to_file(geojson_path, driver="GeoJSON")
                shutil.make_archive(shp_base, "zip", tmp_dir, "detections.geojson")

        with open(out_zip_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

class InferResponse(BaseModel):
    total_weeds: int
    field_area_m2: float
    avg_density: float
    pix_x: list
    pix_y: list
    utm_x: list
    utm_y: list
    lons: list
    lats: list
    meta: dict
    disp_w: int
    disp_h: int
    scale_x: float
    scale_y: float
    counts_base: list   
    x0_m: float
    y0_m: float
    cell_m: int
    thumb_png_b64: str
    shp_zip_b64: str

@app.get("/health")
def health():
    _ = load_model()
    return {"status": "ok"}

@app.post("/infer", response_model=InferResponse)
def infer(file: UploadFile = File(...)):
    src_path = save_upload_to_temp(file)
    try:
        res = run_inference_from_path(src_path, TILE_SIZE)
        meta = res["meta"]

        # Thumbnail in UTM
        thumb_png, (disp_w, disp_h), (scale_x, scale_y) = compute_thumbnail_from_path(
            src_path, MAX_DISPLAY, meta["utm_epsg"]
        )
        thumb_png_b64 = base64.b64encode(thumb_png).decode("utf-8")

        # Shapefile (zipped)
        shp_zip_bytes = build_shapefile_zip(res["lons"], res["lats"])
        shp_zip_b64 = base64.b64encode(shp_zip_bytes).decode("utf-8")

        # Counts base grid
        counts_base, x0_m, y0_m, cell_m = build_counts_safe(
            res["utm_x"], res["utm_y"], meta["utm_bounds"]
        )

        payload = {
            "total_weeds": res["total_weeds"],
            "field_area_m2": res["field_area_m2"],
            "avg_density": res["avg_density"],
            "pix_x": res["pix_x"],
            "pix_y": res["pix_y"],
            "utm_x": res["utm_x"],
            "utm_y": res["utm_y"],
            "lons": res["lons"],
            "lats": res["lats"],
            "meta": meta,
            "disp_w": int(disp_w),
            "disp_h": int(disp_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "counts_base": counts_base.astype(int).tolist(),
            "x0_m": float(x0_m),
            "y0_m": float(y0_m),
            "cell_m": int(cell_m),
            "thumb_png_b64": thumb_png_b64,
            "shp_zip_b64": shp_zip_b64,
        }
        return payload
    finally:
        try:
            os.unlink(src_path)
        except Exception:
            pass
