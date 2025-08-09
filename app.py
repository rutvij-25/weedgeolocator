import streamlit as st
import rasterio
from rasterio.plot import show
from tempfile import NamedTemporaryFile, TemporaryDirectory
from tempfile import mkdtemp
import base64
import matplotlib.pyplot as plt
import pyproj
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from ultralytics import YOLO
import pyproj
import zipfile
import geopandas as gpd
import shutil
import io

# Define the projection objects for EPSG:32618 and WGS84
proj_UTM = pyproj.Proj(init='epsg:32618')  # UTM zone 18N
proj_WGS84 = pyproj.Proj(init='epsg:4326')  # WGS84
wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
model = YOLO('best.pt')

def make_tiles(image):
    width, height = image.size

    # Define tile size
    tile_size = 640

    tiles = {'img':[], 'coords':[], 'result':[]}

    # Iterate over the image in tiles
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Calculate coordinates
            left = x
            upper = y
            right = min(x + tile_size, width)
            lower = min(y + tile_size, height)

            # Crop the tile
            tile = image.crop((left, upper, right, lower))

            tiles['img'].append(tile)
            tiles['coords'].append((left, upper, right, lower))
            tiles['result'].append(model(tile))
            # Save the tile as PNG with coordinates in the filename
            # tile.save(f"tiles/tile_{left}_{upper}_{right}_{lower}.png")

            # Optionally, you can also keep track of the tile's position
            # print(f"tiles/Tile ({left},{upper}) to ({right},{lower}) saved and weeds are detected")

    return tiles
def zip_file(file_to_zip, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        zipf.write(file_to_zip, arcname=file_to_zip)

def make_orthodict(tiles,r):
    orthodict = {}
    i = 0
    for idx, res in enumerate(tiles['result']):
    
        if(0 in res[0].boxes.cls.tolist()):
            temp_dict = {}
            ids = [id for id, v in enumerate(res[0].boxes.cls.tolist()) if v == 0]
            x, y = tiles['coords'][idx][0], tiles['coords'][idx][1]
            # print('x,y',x, y)
            realx = []
            realy = []
            cxs = []
            cys = []
            mapx = []
            mapy = []
            idlist = []
            for xywh in np.array(res[0].boxes.xywh.tolist())[ids,:]:
                cx, cy = xywh[0], xywh[1]
                # print('cx, cy',cx, cy)
                # print('xr, yr',x + cx, y + cy)
                f = [x + cx, y + cy][::-1]
                mx, my = r.xy(*f)
                realx.extend([x + cx])
                realy.extend([y + cy])
                cxs.extend([cx])
                cys.extend([cy])
                mapx.extend([mx])
                mapy.extend([my])
                idlist.append(i)
                i = i + 1
            temp_dict['orig_img'] = tiles['img'][idx]
            temp_dict['tagged_img'] = res[0].plot(conf=False)[...,::-1]
            temp_dict['realcoords'] = tiles['coords'][idx]
            temp_dict['centersx'] = cxs
            temp_dict['centersy'] = cys
            temp_dict['transformedcentersx'] = realx
            temp_dict['transformedcentersy'] = realy
            temp_dict['mapcentersx'] = mapx
            temp_dict['mapcentersy'] = mapy
            temp_dict['ids'] = idlist
            orthodict[idx] = temp_dict
    return orthodict

    # Convert image data to Base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.markdown("""
### 📍 Weed Detection and Geolocation App

This application allows you to upload a georeferenced **orthomosaic TIFF image** (e.g., from a drone),
automatically detect **weeds** using a pre-trained **YOLO model**, and generate a shapefile with precise GPS locations of the detections.

**Key Features:**
- Upload large `.tif` orthomosaic imagery
- Detect weeds using tile-wise YOLO inference
- Convert image coordinates to GPS (WGS84)
- Download a ready-to-use **.shp file** with detections
- View an interactive visualization of weed locations

This tool is designed for **precision agriculture**, especially useful for **spot spraying** and **weed mapping** workflows.

""")


uploaded_file = st.file_uploader("Upload TIFF file", type=["tif", "tiff"])

if uploaded_file is not None:

    tmp = NamedTemporaryFile(delete=False, suffix=".tif")
    tmp.write(uploaded_file.getvalue())
    tmp_path = tmp.name

    src = rasterio.open(tmp_path)
    im = Image.open(tmp_path)

    if src.crs:
        st.write(f"CRS: {src.crs}")
        st.write("Georeferenced: Yes")
    else:
        st.write("Georeferenced: No")
   
    tiles = make_tiles(im)
            
    orthodict = make_orthodict(tiles,src)

    IDLIST = []
    REALX = []
    REALY = []
    for key in orthodict.keys():
        REALX.extend(orthodict[key]['transformedcentersx'])
        REALY.extend(orthodict[key]['transformedcentersy'])
        IDLIST.extend(orthodict[key]['ids'])
                    
    ID = []
    MCX = []
    MCY = []
    LAT = []
    LON = []

    for key in orthodict.keys():
        ID.extend(orthodict[key]['ids'])
        for mx, my in zip(orthodict[key]['mapcentersx'], orthodict[key]['mapcentersy']):
            lon, lat = pyproj.transform(proj_UTM, proj_WGS84, mx, my)
            LON.append(lon)
            LAT.append(lat)    

    cdict = {'id':ID, 'lat':LAT, 'lon':LON}
    cdf = pd.DataFrame(cdict)
    cgdf = gpd.GeoDataFrame(cdf, geometry = gpd.points_from_xy(cdf['lon'], cdf['lat']))
    temp_dir = mkdtemp()

# Save GeoDataFrame to shapefile
    cgdf.to_file(filename=f'{temp_dir}/shapefile', driver='ESRI Shapefile',  crs = wkt)

    # Zip the shapefile
    shutil.make_archive(f'{temp_dir}/shapefile', 'zip', temp_dir, 'shapefile')

    # Define file path and name
    file_path = f'{temp_dir}/shapefile.zip'
    file_name = 'shapefile.zip'

    # Create download button
    st.download_button(
    label="Download .shp file",
    data=open(file_path, 'rb').read(),
    file_name=file_name,
    mime='application/zip'
    )

    st.subheader("Visualizing the orthomosaic")
    st.write(f"{len(REALX)} weeds detected")
 
    # Resizing the image
    im_resized = im.resize((8192, 8192))
    fig = px.imshow(im_resized)
    # Calculate the scaling factors
    scale_x = im_resized.width / im.width
    scale_y = im_resized.height / im.height

# Scale the coordinates
    scaled_x = [x * scale_x for x in REALX]
    scaled_y = [y * scale_y for y in REALY]

    scatter_trace = go.Scatter(x=scaled_x, y=scaled_y, mode='markers', marker=dict(color="red", size=2))

    fig.add_trace(scatter_trace)

    # Display the plot
    st.plotly_chart(fig)





 







