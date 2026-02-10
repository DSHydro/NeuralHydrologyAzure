"""
Prepare Data for NeuralHydrology Training

This script prepares the input data required by the NeuralHydrology framework. It performs
the following tasks:

1. Data Organization:
   - Creates the required directory structure: data/time_series and data/attributes
   - Processes streamflow data from train/validation/test splits
   - Processes meteorological forcing data from CARRA reanalysis
   - Processes static catchment attributes

2. Static Attributes Processing:
   - Reads catchment attributes from Catchment_attributes.csv
   - Selects and processes required attributes:
     * elev_mean: Mean elevation (m)
     * slope_mean: Catchment slope (m/km)
     * aspect: Converted to sin/cos components
     * p_mean: Mean daily precipitation (mm/day)
     * frac_snow: Snow fraction (-)
     * ndvi_max: Maximum NDVI (-)
     * g_frac: Glacier fraction (-)
   - Checks for missing values
   - Saves processed attributes to data/attributes/attributes.csv

3. Time Series Processing (for each basin):
   - Reads meteorological forcing data
   - Processes wind components (converts speed/direction to u/v components)
   - Renames variables to match NeuralHydrology requirements
   - Converts streamflow from m³/s to mm/day
   - Creates a standardized time index (1999-10-01 to 2024-09-30)
   - Saves data as NetCDF files in data/time_series/

4. Additional Tasks:
   - Creates basin list files for train/val/test splits
   - Performs error checking and reporting
   - Handles missing data appropriately

Input Requirements:
- Streamflow data in processed_data (train_data.csv, validation_data.csv, test_data.csv)
- Meteorological data in CARRA format
- Catchment attributes file with required static properties

Output Structure:
- filtered_data/
  ├── time_series/          # NetCDF files with time series data
  └── attributes/           # CSV file with static attributes

Author: Hernán Querbes
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

# Define paths
# base_dir = Path("/inputs")
met_dir = Path("/inputs/data_updated_2/time_series")
area_file= Path("/inputs/data_updated_2/attributes/attributes_other.csv")
attributes_files= Path("/inputs/data_updated_2/attributes")

# Define output structure required by neuralhydrology
data_dir = Path("./preparing_data/filtered_data")
time_series_dir = data_dir / "time_series"
attributes_dir = data_dir / "attributes"

# Create output directories
data_dir.mkdir(exist_ok=True)
time_series_dir.mkdir(exist_ok=True)
attributes_dir.mkdir(exist_ok=True)

# Define variable mapping from CAMELS_UY names to neuralhydrology names
variable_mapping = {
    'temperature_2m_min': 'tmin_C',
    'temperature_2m_max': 'tmax_C',
    'surface_net_solar_radiation_mean': 'srad_W_m2',
    'total_precipitation_sum': 'prcp_mm_day'
}

# Read streamflow data
train_data = pd.read_csv("./preparing_data/processed_data/highqual_with_short_estimated/train_data.csv", index_col=0, parse_dates=True)
val_data = pd.read_csv("./preparing_data/processed_data/highqual_with_short_estimated/validation_data.csv", index_col=0, parse_dates=True)
test_data = pd.read_csv("./preparing_data/processed_data/highqual_with_short_estimated/test_data.csv", index_col=0, parse_dates=True)

# Combine all streamflow data
streamflow_data = pd.concat([train_data, val_data, test_data]).sort_index()

# Get list of basins
basins = streamflow_data.columns.tolist()

basins_mapping = {
    'Paso Mazangano': 'CAMELS_UY_10',
    'Paso de Coelho': 'CAMELS_UY_7',
    'Sarandi del Yi': 'CAMELS_UY_12',
    'Paso de las Toscas': 'CAMELS_UY_8',
    'Paso de las Piedras (R3)': 'CAMELS_UY_15',
    'Paso del Borracho': 'CAMELS_UY_6',
    'Bequelo': 'CAMELS_UY_16',
    'Paso de las Piedras': 'CAMELS_UY_2',
    'Paso Baltasar': 'CAMELS_UY_5',
    'Fraile Muerto': 'CAMELS_UY_11',
    'Paso de los Mellizos': 'CAMELS_UY_14',
    'Paso Manuel Diaz': 'CAMELS_UY_3',
    'Paso Aguiar': 'CAMELS_UY_9',
    'Paso de la Compania': 'CAMELS_UY_1',
    'Tacuarembo': 'CAMELS_UY_4',
    'Durazno': 'CAMELS_UY_13'
}

# Read catchment attributes
# attrs_df = pd.read_csv(attributes_file, delimiter=';')
# attrs_df['id'] = attrs_df['id'].astype(str)  # Convert IDs to string for consistency
# attrs_df = attrs_df.set_index('id')
df = pd.read_csv(attributes_files / "attributes_other.csv")

df = df.merge(
    pd.read_csv(attributes_files / "attributes_caravan.csv"),
    on="gauge_id",
    how="outer"
)

df = df.merge(
    pd.read_csv(attributes_files / "attributes_hydroatlas.csv"),
    on="gauge_id",
    how="outer"
)

# Set gauge_id as the index
attrs_df = df.set_index("gauge_id")

# Select only the basins we want to use
target_gauges = list(basins_mapping.values())
attrs_df = attrs_df.loc[target_gauges]

# Define required attributes
attributes_mapping = {
    'gauge_name': 'gauge_name',
    'ele_mt_sav': 'elev_mean',
    'slp_dg_sav': 'slope_mean',
    'p_mean': 'p_mean',
    'area': 'area_gages2',
    'frac_snow': 'frac_snow',
    'aridity_FAO_PM': 'aridity',
    'pet_mean_FAO_PM': 'pet_mean',
    'high_prec_freq': 'high_prec_freq',
    'high_prec_dur': 'high_prec_dur',
    'low_prec_freq': 'low_prec_freq',
    'low_prec_dur': 'low_prec_dur',
    'snd_pc_sav': 'sand_frac',
    'slt_pc_sav': 'silt_frac',
    'cly_pc_sav': 'clay_frac'
}

attributes_to_keep = {
    'gauge_name',
    'ele_mt_sav',    # Mean elevation (m)
    'slp_dg_sav',   # Catchment slope (m/km)
    'p_mean',       # Mean daily precipitation (mm/day)
    'area',
    'frac_snow',    # Snow fraction (-)
    'aridity_FAO_PM',
    'pet_mean_FAO_PM',
    'high_prec_freq',
    'high_prec_dur',
    'low_prec_freq',
    'low_prec_dur',
    'snd_pc_sav',
    'slt_pc_sav',
    'cly_pc_sav'
}

# # Define required attributes
# required_attributes = {
#     'elev_mean',    # Mean elevation (m)
#     'slope_mean',   # Catchment slope (m/km)
#     'asp_mean',     # Aspect (degrees) - will be converted to sin/cos
#     'p_mean',       # Mean daily precipitation (mm/day)
#     'frac_snow',    # Snow fraction (-)
#     'ndvi_max',     # Maximum NDVI (-)
#     'g_frac'        # Glacier fraction (-)
# }

# Select and process static attributes
final_attrs = attrs_df[list(attributes_to_keep)]
final_attrs = final_attrs.rename(columns=attributes_mapping)

# Check for any missing values in attributes
missing_values = final_attrs.isnull().sum()
if missing_values.any():
    print("\nERROR: Found missing values in attributes:")
    print(missing_values[missing_values > 0])
    print("\nPlease fix the missing values in the source data before proceeding.")
    exit(1)

# # Basic validation of attribute ranges
# print("\nValidating attribute ranges...")
# validations = {
#     'elev_mean': (0, 3000),      # Elevation range for Iceland (m)
#     'slope_mean': (0, 1000),     # Slope in m/km
#     'aspect_sin': (-1, 1),       # Sine component
#     'aspect_cos': (-1, 1),       # Cosine component
#     'p_mean': (0, 50),           # Daily precipitation in mm/day
#     'frac_snow': (0, 1),         # Snow fraction (dimensionless)
#     'ndvi_max': (-1, 1),         # NDVI range (dimensionless)
#     'g_frac': (0, 1)             # Glacier fraction (dimensionless)
# }

# for attr, (min_val, max_val) in validations.items():
#     values = final_attrs[attr]
#     if (values < min_val).any() or (values > max_val).any():
#         print(f"\nWARNING: {attr} has values outside expected range [{min_val}, {max_val}]")
#         print(f"Min: {values.min():.2f}, Max: {values.max():.2f}")

# # Print attribute statistics and correlations
# print("\nAttribute statistics:")
# print(final_attrs.describe())
# print("\nAttribute correlations:")
# print(final_attrs.corr().round(2))

# Save attributes file
final_attrs.to_csv(attributes_dir / "attributes.csv")
print(f"\nSaved static attributes for {len(final_attrs)} basins")

# Create a standard time index for the period we want
start_date = pd.Timestamp('1989-01-01') #('1989-09-01')  #('1999-10-01')
end_date = pd.Timestamp('2019-12-31') #('2009-08-31')  #('2024-09-30')
date_index = pd.date_range(start=start_date, end=end_date, freq='D', name='date')
print(f"\nUsing time period: {date_index[0]} to {date_index[-1]} ({len(date_index)} days)")

# Process each basin's time series data
print(f"\nProcessing {len(basins)} basins...")

areas= pd.read_csv(area_file)
areas = areas.set_index('gauge_id')

for basin in basins:
    basin_id=basins_mapping[basin]
    print(f"\nProcessing basin {basin}")
    
    # Read meteorological data
    met_file = met_dir / f"{basin_id}.nc"
        
    try:
        # Read meteorological data
        # met_df = pd.read_csv(met_file, delimiter=';')
        # Read meteorological data (xarray Dataset)
        met_ds = xr.open_dataset(met_file)

        # Check for required variables
        required_vars = list(variable_mapping.keys())
        available_vars = set(met_ds.data_vars)
        
        missing_vars = [v for v in required_vars if v not in available_vars]
        if missing_vars:
            print(f"Warning: Missing variables in meteorological data for {basin}: {missing_vars}")
            print("Available variables:", list(available_vars))
            continue
        
        # Select only variables we need
        met_ds = met_ds[required_vars]
        
        # Rename using your mapping
        met_ds = met_ds.rename(variable_mapping)
        
        # Convert xarray → pandas using existing datetime coordinate
        met_df = met_ds.to_dataframe()
    
        # Ensure the index is named "date"
        met_df.index.name = "date"
        
        # Keep only final variables you want
        variables_to_keep = [
            "tmin_C",
            "tmax_C",
            "srad_W_m2",
            "prcp_mm_day"
        ]
        
        met_df = met_df[variables_to_keep]
        
        # Get streamflow data and convert from m³/s to mm/day
        qobs = streamflow_data[basin]    
        area_m2 = areas.loc[basin_id, 'area'] * 1e6  # Convert km² to m²
        
        qobs_mmday = (qobs * 86400 * 1000) / area_m2  # Convert m³/s to mm/day

        # Combine all variables into a single DataFrame
        met_df['QObs_mm_d'] = qobs_mmday
        
        # Reindex to standard date range, filling gaps with NaN
        met_df = met_df.reindex(date_index)
        
        # Convert to xarray Dataset
        ds = met_df.to_xarray()
        
        # Save to netCDF
        encoding = {var: {'_FillValue': np.nan} for var in ds.data_vars}
        encoding['date'] = {
            'dtype': 'float64',
            'calendar': 'proleptic_gregorian',
            'units': f'days since {start_date.strftime("%Y-%m-%d")}'
        }
        
        ds.to_netcdf(
            time_series_dir / f"{basin_id}.nc",
            encoding=encoding
        )
        
        print(f"Saved time series with {len(ds.data_vars)} variables")
        
    except FileNotFoundError:
        print(f"Warning: No meteorological data found for basin {basin_id}")
        continue
    except Exception as e:
        print(f"Error processing basin {basin_id}: {str(e)}")
        continue

print("\nData preparation complete!")
# print(f"Output structure:")
# print(f"- {time_series_dir}: NetCDF files with time series data")
# print(f"- {attributes_dir}: CSV file with static attributes")

# # Create basin list files (all basins in each list for now)
# for split in ['train', 'val', 'test']:
#     with open(data_dir / f"basins_{split}.txt", 'w') as f:
#         f.write('\n'.join(basins))

# print("\nCreated basin list files (all basins in each list)")
# print("You may want to modify these files to create proper train/val/test splits")

# # Print summary of processed variables
# print("\nProcessed Variables Summary:")
# print("\nMeteorological variables:")
# print("- Wind components (converted from speed/direction):")
# print("  * wind_u_at_10m_agl (zonal wind, positive eastward)")
# print("  * wind_v_at_10m_agl (meridional wind, positive northward)")
# print("\nOther variables:")
# for old_name, new_name in variable_mapping.items():
#     if 'wind' not in old_name:  # Skip wind variables as they're handled differently
#         print(f"- {old_name} -> {new_name}")
# print("\nStreamflow:")
# print("- Converted from m³/s to mm/day using catchment areas") 