# Weed Detection and Geolocation App

1)A **Streamlit**-based web application for detecting and geolocating weeds in **georeferenced orthomosaic GeoTIFF images** (e.g., from drone surveys).  
2)The app uses a **YOLOv8** model to process large images tile-by-tile, extract weed locations, and export a **Shapefile** with precise GPS coordinates.  
3)This tool is designed for **precision agriculture**, enabling farmers and researchers to map weeds for **spot spraying** and **field analysis**.

<video src="https://github.com/user-attachments/assets/0840c78a-67aa-4a64-93ea-2e70dde0fe95"
       controls playsinline muted loop style="max-width:100%; height:auto;">
</video>

---

## Features
- **Upload** `.tif` / `.tiff` georeferenced orthomosaic imagery
- **YOLOv8-based detection** on tiled images for large raster handling
- **Automatic GPS conversion** from UTM to WGS84
- **Export Shapefile** of detections for use in GIS software
- **Interactive visualization** overlaying detections on the orthomosaic and important KPIs
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

### Running with Docker
Build an image
```bash
docker build -t weed-geo-app .
```

Run 
```bash
docker run -p 8501:8501 weed-geo-app
```

