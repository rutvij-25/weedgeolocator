FROM python:3.9-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev proj-bin proj-data build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip "setuptools<81" wheel

ENV GDAL_CONFIG=/usr/bin/gdal-config \
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

RUN printf "numpy==1.26.4\nrasterio==1.3.9\npyogrio==0.11.1\nopencv-python-headless==4.9.0.80\n" > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "numpy==1.26.4"

RUN pip install --no-cache-dir --no-binary rasterio "rasterio==1.3.9"

COPY requirements.txt .
RUN grep -Ev '^(opencv-python|opencv-python-headless|pyogrio|fiona)(==|<=|>=|~=|=|<|>|!=)' requirements.txt > /tmp/req-no-drivers.txt \
 && pip install --no-cache-dir -r /tmp/req-no-drivers.txt

RUN pip install --no-cache-dir --no-binary pyogrio "pyogrio==0.11.1"

RUN pip install --no-cache-dir "opencv-python-headless==4.9.0.80"

FROM python:3.9-slim
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ULTRALYTICS_RUNS_DIR=/tmp/runs \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    GDAL_CACHEMAX=512

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin proj-bin proj-data \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

WORKDIR /app
COPY app.py .
COPY best_STM.pt .

RUN mkdir -p /tmp/ultralytics /tmp/runs /tmp/datasets /tmp/weights && \
    chmod -R 777 /tmp/ultralytics /tmp/runs /tmp/datasets /tmp/weights

RUN python - <<'PY'
import os, json
cfg = os.environ.get("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.makedirs(cfg, exist_ok=True)
os.makedirs(os.environ.get("ULTRALYTICS_RUNS_DIR", "/tmp/runs"), exist_ok=True)
os.makedirs("/tmp/datasets", exist_ok=True)
os.makedirs("/tmp/weights", exist_ok=True)
settings = {
    "settings_version": "0.0.6",
    "runs_dir": os.environ.get("ULTRALYTICS_RUNS_DIR", "/tmp/runs"),
    "datasets_dir": "/tmp/datasets",
    "weights_dir": "/tmp/weights",
    "sync": False,
    "uuid": "",
    "clearml": False,
    "comet": False,
    "dvc": False,
    "hub": False,
    "mlflow": False,
    "neptune": False,
    "raytune": False,
    "tensorboard": False,
    "wandb": False,
}
with open(os.path.join(cfg, "settings.json"), "w") as f:
    json.dump(settings, f, indent=2)
print("Preseeded Ultralytics settings at", os.path.join(cfg, "settings.json"))
PY

RUN mkdir -p /app/.streamlit && \
    printf "[server]\nheadless = true\nfileWatcherType = \"none\"\n\n[browser]\ngatherUsageStats = false\n" > /app/.streamlit/config.toml

RUN python - <<'PY'
import numpy, rasterio, cv2, json, os
print("NumPy:", numpy.__version__)
print("Rasterio:", rasterio.__version__)
print("OpenCV:", cv2.__version__)
print("YOLO settings exist:", os.path.exists(os.environ["YOLO_CONFIG_DIR"] + "/settings.json"))
with open(os.environ["YOLO_CONFIG_DIR"] + "/settings.json", "r") as f:
    print("YOLO settings content:", json.load(f))
PY

RUN python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"

EXPOSE 8080

RUN mkdir -p /app/.streamlit && \
    printf "[server]\nheadless = true\nfileWatcherType = \"none\"\nmaxUploadSize = 10240\n\n[browser]\ngatherUsageStats = false\n" > /app/.streamlit/config.toml

CMD ["python","-m","streamlit","run","app.py", "--server.port=8080","--server.address=0.0.0.0","--server.maxUploadSize=10240"]
