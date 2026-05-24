# 1_uy_basins_data

This folder prepares the complete input dataset for NeuralHydrology training on Uruguayan river basins. It processes raw streamflow records, assembles catchment attributes and meteorological time series, and extracts basin-averaged precipitation from two gridded products (CHIRPS and MSWEP). The final step merges everything into a single set of NetCDF files ready for model training.

## Notebooks

The six notebooks are intended to be run **in order**. Each one produces outputs that the next one consumes.

### `1_streamflow_data_selection.ipynb`

Reads raw daily streamflow Excel files for 16 CAMELS-UY basins and applies quality-based filtering to produce clean train/validation/test splits.

**Key steps:**
- Loads raw gauge records and maps station names to CAMELS-UY IDs
- Handles estimated-data quality codes (`Dato estimado o calculado`): keeps consecutive estimated periods of ≤ 10 days and replaces longer ones with NaN
- Generates quality-validation plots per gauge (streamflow + color-coded quality bar)
- Splits the full record into three periods:
  - **Validation:** 1989-10-01 – 1999-09-30
  - **Training:** 1999-10-01 – 2008-09-30
  - **Test:** 2008-10-01 – 2019-12-31

**Outputs** → `./processed_data_2/highqual_with_short_estimated/`
- `train_data.csv`, `validation_data.csv`, `test_data.csv`
- Per-gauge validation plots in `streamflow_series/` and `data_validation/`

---

### `2_prepare_Catchment_attributes_and_timeseries.ipynb`

Assembles the NeuralHydrology-format dataset by combining meteorological time series, streamflow, and static catchment attributes for the 11 basins that pass data quality thresholds.

**Key steps:**
- Reads ERA5-based NetCDF meteorological files and renames variables to NeuralHydrology conventions (e.g. `total_precipitation_sum` → `prcp_mm_day`, `temperature_2m_min` → `tmin_C`)
- Converts streamflow from m³/s to mm/day using each basin's drainage area
- Merges static attributes from three CAMELS-UY attribute files (`attributes_other.csv`, `attributes_caravan.csv`, `attributes_hydroatlas.csv`), retaining 15 catchment descriptors (elevation, slope, aridity, soil texture fractions, etc.)
- Aligns all variables to a standard daily index spanning 1989-01-01 – 2019-12-31 (11 322 days)

**Outputs** → `./filtered_data_2/`
- `time_series/CAMELS_UY_*.nc` — one NetCDF per basin with variables: `tmin_C`, `tmax_C`, `srad_W_m2`, `prcp_mm_day`, `QObs_mm_d`
- `attributes/attributes.csv` — static attributes table (16 basins × 15 variables)

---

### `3_filling_missing_precip.ipynb`

Fills gaps in the ground-based gauge precipitation records using a nearest-neighbour linear regression approach before those records are used as an additional precipitation product.

**Key steps:**
- Loads watershed shapefiles and plots basin boundaries against gauge network locations
- Computes pairwise geodesic distances between all precipitation gauges
- For each station with missing values, identifies the nearest available gauge, fits a linear regression on overlapping dates, and uses it to fill gaps
- Saves filled time series to Excel and then aligns them to the standard CAMELS-UY time index

**Outputs** → `./gauge_data/`
- `filled_precip/<station>_filled_precip.xlsx` — gap-filled daily records per gauge
- `gauge_precip_timeseries/CAMELS_UY_*_precip.csv` — basin-level gauge precipitation CSVs aligned to the model time window

---

### `4_chirps_precip_timeseries.ipynb`

Extracts daily basin-averaged precipitation from the **CHIRPS v3** gridded dataset for all 16 CAMELS-UY basins.

**Key steps:**
- Reads daily CHIRPS GeoTIFF files from `/inputs/CHIRPS_v3/`
- Loads watershed shapefiles and reprojects to EPSG:4326
- Uses `exactextract` to compute the area-weighted spatial mean over each basin polygon for every daily raster
- Masks no-data values (−9999) before extraction

**Outputs** → `./chirps_precip_timeseries/`
- `CAMELS_UY_*_precip.csv` — one CSV per basin with a daily `precipitation` column

---

### `5_mswep_precip_timeseries.ipynb`

Extracts daily basin-averaged precipitation from the **MSWEP V2.8** gridded dataset for all 16 CAMELS-UY basins, using parallel processing to handle the large daily NetCDF archive efficiently.

**Key steps:**
- Reads daily MSWEP NetCDF files from `/inputs/MSWEP_V280/Past/Daily/` for 1989–2019
- Loads and combines all watershed shapefiles into a single GeoDataFrame
- Renames spatial dimensions (`lat`/`lon` → `y`/`x`) and writes CRS before extraction
- Processes files in parallel using `ThreadPoolExecutor` (4 workers) with `exactextract` for area-weighted spatial means

**Outputs** → `./mswep_precip_timeseries/`
- `CAMELS_UY_*_precip.csv` — one CSV per basin with a daily `precip_mm` column

---

### `6_keeping_all_precip_products.ipynb`

Merges all three precipitation products (gauge, MSWEP, CHIRPS) into the existing CAMELS-UY NetCDF files, producing the final dataset used for NeuralHydrology training experiments.

**Key steps:**
- Reads the NetCDF files produced by notebook 2 (from `./filtered_data_2/time_series/`)
- Loads the three precipitation CSVs for each basin and aligns them to the existing time coordinate
- Appends three new variables to each dataset: `prcp_gauge_mm_day`, `prcp_mswep_mm_day`, `prcp_chirps_mm_day`
- Copies the attributes directory unchanged

**Outputs** → `./data/`
- `time_series/CAMELS_UY_*.nc` — final NetCDF files with 7 variables (original 5 + 3 precipitation products)
- `attributes/` — copy of the static attributes from `filtered_data_2/`

---

## Data Flow Summary

```
Raw XLS streamflow files
        │
        ▼
1_streamflow_data_selection  →  processed_data_2/ (train/val/test CSVs)
        │
        ▼
2_prepare_catchment_attributes  →  filtered_data_2/ (NetCDF + attributes)
        │
  ┌─────┴──────┐
  ▼            ▼
3_filling_missing_precip    4_chirps_precip_timeseries    5_mswep_precip_timeseries
  │ (gauge CSVs)              │ (CHIRPS CSVs)               │ (MSWEP CSVs)
  └──────────────────────────┼─────────────────────────────┘
                             ▼
               6_keeping_all_precip_products  →  data/ (final NetCDF + attributes)
```

## Dependencies

```
pandas, numpy, xarray, geopandas, rioxarray, exactextract,
matplotlib, contextily, sklearn, pyproj, tqdm
```

Install the `neuralhydrology` conda environment (Python 3.10) before running these notebooks.

## Input Data

| Source | Location | Description |
|--------|----------|-------------|
| CAMELS-UY raw streamflow | `./streamflow_timeseries/` | Daily gauge records (XLS) |
| CAMELS-UY meteorology + attributes | `/inputs/data_updated_2/` | ERA5-derived NetCDF + attribute CSVs |
| Watershed shapefiles | `./watersheds/shapefile/` | One ZIP per basin (EPSG:4326) |
| Gauge precipitation records | `./gauge_data/precip_uy/` | Raw daily precip spreadsheets |
| CHIRPS v3 daily rasters | `/inputs/CHIRPS_v3/` | GeoTIFF files |
| MSWEP V2.8 daily grids | `/inputs/MSWEP_V280/Past/Daily/` | NetCDF files named `YYYYDDD.nc` |
