# 2_us_127_basins

This folder prepares the multi-precipitation-product dataset for 127 US CAMELS basins. It extracts basin-averaged precipitation from two gridded datasets (CHIRPS v3 and MSWEP V2.8) and then assembles extended NetCDF time series that combine the original CARAVAN meteorological and streamflow data with ERA5-Land, CAMELS Daymet, CHIRPS, and MSWEP precipitation — producing the final inputs for NeuralHydrology training experiments.

## Notebooks

Run the notebooks **in order**. Notebooks 1 and 2 can be run in parallel since they are independent of each other; notebook 3 depends on both.

---

### `1_downloading_chirps_v3.ipynb`

Extracts daily basin-averaged precipitation from the **CHIRPS v3** gridded dataset for the subset of CAMELS basins listed in `basins_subset_test.txt`.

**Outputs** → `./chirps_precip_timeseries/`
- `<gauge_id>_precip.csv` — daily `precipitation` time series per basin

---

### `2_mswep_weighted_precip_timeseries.ipynb`

Extracts daily basin-averaged precipitation from the **MSWEP V2.8** gridded dataset for the same basin subset, covering 1989–2019.

**Outputs** → `./mswep_precip_timeseries/`
- `<gauge_id>_precip.csv` — daily `precipitation` time series per basin

---

### `3_creating_extended_datasets.ipynb`

Assembles the final multi-product NetCDF dataset by merging the original CARAVAN data with ERA5-Land (from the CARAVAN multi-met Zarr store), CAMELS Daymet, CHIRPS v3, and MSWEP precipitation for each basin.

**Key steps:**
- Connects to the **CARAVAN multi-met Google Cloud Zarr store** (`gs://caravan-multimet/v1.1/`) and explores available products: CPC, IMERG, CHIRPS, ERA5-Land, CHIRPS-GEFS, HRES, and GraphCast
- For each basin, selects ERA5-Land `era5land_total_precipitation` over 1989-01-01 – 2019-12-31 and merges it with the original CARAVAN variables (`total_precipitation_sum`, `temperature_2m_max`, `temperature_2m_min`, `surface_net_solar_radiation_mean`, `streamflow`)
- Sequentially appends three more precipitation variables by loading and aligning to the existing time coordinate:
  - `camels_precipitation` — from CAMELS Daymet basin-mean forcing TXT files
  - `chirps_precipitation` — from the CSVs produced by notebook 1
  - `mswep_precipitation` — from the CSVs produced by notebook 2

**Outputs** → `./data/time_series/`
- `<gauge_id>.nc` — one NetCDF per basin with 8 variables: `streamflow`, `total_precipitation_sum` (ERA5), `temperature_2m_max`, `temperature_2m_min`, `surface_net_solar_radiation_mean`, `era5land_total_precipitation`, `camels_precipitation`, `chirps_precipitation`, `mswep_precipitation`
