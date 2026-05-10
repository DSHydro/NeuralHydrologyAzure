import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from exactextract import exact_extract

# shapefiles folder
path_shapefiles = Path('../MSWEP/shapefiles')
# CHIRPS folder
chirps_folder = Path('/inputs/CHIRPS_v3/')
# Output directory
output_dir = Path('./precip_timeseries')
output_dir.mkdir(exist_ok=True)

# Mapping shapefile names to CAMELS_UY IDs
basins_mapping = {
    'paso_mazangano': 'CAMELS_UY_10',
    'picada_de_coelho': 'CAMELS_UY_7',
    'sarandi_del_yi': 'CAMELS_UY_12',
    'paso_de_las_toscas': 'CAMELS_UY_8',
    'paso_de_las_piedras_rn': 'CAMELS_UY_15',
    'paso_del_borracho': 'CAMELS_UY_6',
    'bequelo': 'CAMELS_UY_16',
    'paso_de_las_piedras': 'CAMELS_UY_2',
    'paso_baltasar': 'CAMELS_UY_5',
    'fraile_muerto': 'CAMELS_UY_11',
    'paso_de_los_mellizos': 'CAMELS_UY_14',
    'paso_manuel_diaz': 'CAMELS_UY_3',
    'paso_aguiar': 'CAMELS_UY_9',
    'paso_de_la_compania': 'CAMELS_UY_1',
    'tacuarembo': 'CAMELS_UY_4',
    'durazno': 'CAMELS_UY_13'
}

# List all shapefiles and load into a combined GeoDataFrame
shapefiles = [f for f in os.listdir(path_shapefiles) if f.endswith('.zip')]
gdfs = []
for shp_file in shapefiles:
    basin_name = shp_file.replace('.zip', '')
    if basin_name not in basins_mapping:
        continue
    gdf = gpd.read_file(path_shapefiles / shp_file).to_crs("EPSG:4326")
    gdf["basin_name"] = basin_name
    gdfs.append(gdf[["basin_name", "geometry"]])

all_basins_gdf = pd.concat(gdfs, ignore_index=True)
basin_names = all_basins_gdf["basin_name"].tolist()

# List daily files
chirps_files = sorted(chirps_folder.glob("*.tif"))

# Accumulator
records = {name: [] for name in basin_names}

# Single loop over files, all basins extracted at once
for file in chirps_files:
    date_str = file.stem.split('.')[-3:]
    date = pd.to_datetime('-'.join(date_str))
    print(f"Processing {date.date()}")

    ds = xr.open_dataset(file)
    precip = ds["band_data"].isel(band=0)
    precip = precip.where(precip != -9999, np.nan)
    precip = precip.rio.write_crs("EPSG:4326")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        precip.rio.to_raster(tmp_path)
        result = exact_extract(tmp_path, all_basins_gdf, ["mean"], output="pandas")
        for i, basin_name in enumerate(basin_names):
            records[basin_name].append((date, result["mean"].iloc[i]))
    finally:
        os.unlink(tmp_path)

    ds.close()

# Save one CSV per basin
for basin_name, ts in records.items():
    camels_id = basins_mapping[basin_name]
    full_ts = pd.Series(
        data=[v for _, v in ts],
        index=[d for d, _ in ts],
        name="precipitation"
    ).sort_index()
    output_file = output_dir / f"{camels_id}_precip.csv"
    full_ts.to_csv(output_file)
    print(f"Saved {output_file}")
