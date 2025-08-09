# Weed Detection and Geolocation App

A **Streamlit**-based web application for detecting and geolocating weeds in **georeferenced orthomosaic GeoTIFF images** (e.g., from drone surveys).  

The app uses a **YOLOv8** model to process large images tile-by-tile, extract weed locations, and export a **Shapefile** with precise GPS coordinates.  
This tool is designed for **precision agriculture**, enabling farmers and researchers to map weeds for **spot spraying** and **field analysis**.

![Weed Detection Demo](assets/websitevideo.gif)
---

## Features
- **Upload** `.tif` / `.tiff` georeferenced orthomosaic imagery
- **YOLOv8-based detection** on tiled images for large raster handling
- **Automatic GPS conversion** from UTM to WGS84
- **Export Shapefile** of detections for use in GIS software
- **Interactive visualization** overlaying detections on the orthomosaic
- Currently supports **grass weeds** only (common ragweed, Palmer amaranth, and common lambsquarters coming soon)

---

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
Then

```bash
streamlit run app.py
```

Running with Docker
```bash
docker build -t weed-geo-app .
```

Run with container
```bash
docker run -p 8501:8501 weed-geo-app
```

