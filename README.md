# Weed Detection and Geolocation App

A **Streamlit**-based web application for detecting and geolocating weeds in **georeferenced orthomosaic TIFF images** (e.g., from drone surveys).  
The app uses a **YOLOv8** model to process large images tile-by-tile, extract weed locations, and export a **shapefile** with precise GPS coordinates.
This tool is designed for **precision agriculture**, enabling farmers and researchers to map weeds for **spot spraying** and **field analysis**.

---

## Features
- **Upload** `.tif` / `.tiff` georeferenced orthomosaic imagery
- **YOLOv8-based detection** on tiled images for large raster handling
- **Automatic GPS conversion** from UTM to WGS84
- **Export shapefile** of detections for use in GIS software
- **Interactive visualization** overlaying detections on the orthomosaic
- **Currenlty supporting grass weeds only (I will add common ragweed, palmer amaranth and common lambsquaters soon)

---

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
