"""
This script processes streamflow data with a modified quality criteria that balances data availability with quality:

Quality Code Handling:
- Quality code "Dato aceptado" (good) is always kept
- Quality code "Dato estimado o calculado" (estimated) is kept only for short periods (≤10 days)
- All other quality codes ("Serie incompleta", "Aproximado por falta de escala", "Afectado por remanso","Dato dudoso") are set to NaN

Some outputs:
   - Shows where short estimated periods were kept
   - Quantifies the difference in data availability

Output files are saved in processed_data/highqual_with_short_estimated/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import groupby
from operator import itemgetter

# Configuration
plot_streamflow_series = True  # Plot individual streamflow time series

# Define base path to LamaH-Ice
base_path = Path("./preparing_data/caudales_filtrados") #caudales_filtrados")

# Define paths relative to base path
path_to_xls_files = base_path #/ "D_gauges/2_timeseries/daily"
# attrs_file_path = base_path / "D_gauges/1_attributes/Gauge_attributes.csv"

#File to match gaude_id with gauge_name
names_path=Path("/inputs/data_updated_2/attributes/attributes_other.csv")
names_file = pd.read_csv(names_path)

# Create directories for processed data and plots
output_dir = Path("./preparing_data/processed_data/highqual_with_short_estimated")
plots_dir = output_dir / "streamflow_series"
# omitted_plots_dir = output_dir / "streamflow_series" / "omitted_gauges"
validation_dir = output_dir / "data_validation"
curated_flows_dir = output_dir / "curated_flows"
output_dir.mkdir(exist_ok=True, parents=True)
plots_dir.mkdir(exist_ok=True, parents=True)
# omitted_plots_dir.mkdir(exist_ok=True, parents=True)
validation_dir.mkdir(exist_ok=True, parents=True)
curated_flows_dir.mkdir(exist_ok=True, parents=True)

def find_long_estimated_periods(quality_series, max_length=10):
    """
    Process quality codes to handle estimated data (code "Dato estimado o calculado"):
    - Keep periods of ≤10 consecutive days
    - Mark longer periods as missing (code 250)
    Returns a Series of processed quality codes.
    """
    # Convert to numpy array for faster processing
    quality_codes = quality_series.values.copy()
    processed_codes = quality_codes.copy()
    
    # Initialize counter for consecutive estimated days
    count = 0
    
    # Process each quality code
    for i in range(len(quality_codes)):
        if quality_codes[i] == "Dato estimado o calculado":
            count += 1
            # When we hit 11 consecutive days
            if count == max_length + 1:
                # Set the previous 10 days to missing
                processed_codes[i-max_length:i] = 250
                # Set current day to missing
                processed_codes[i] = 250
            # For any subsequent days in long periods
            elif count > max_length + 1:
                processed_codes[i] = 250
        else:
            # Reset counter when we see any other quality code
            count = 0
    
    return pd.Series(processed_codes, index=quality_series.index)

def process_quality_codes(df, gauge_id):
    """
    Process quality codes with modified criteria:
    - Keep code "Dato aceptado" (good quality)
    - Keep code "Dato estimado o calculado" (estimated) for periods ≤10 days
    - Set all other data ("Serie incompleta","Aproximado por falta de escala", "Afectado por remanso","Dato dudoso") to missing (code 250)
    """
    # Start with the original quality codes
    processed_qc = df['Índice de calidad'].copy()
    
    # Process estimated data (code 100)
    processed_qc = find_long_estimated_periods(processed_qc)
    
    # Set suspect ("Aproximado por falta de escala", "Afectado por remanso","Dato dudoso") to missing (code 250)
    processed_qc[processed_qc == "Serie incompleta"] = 250
    processed_qc[processed_qc == "Aproximado por falta de escala"] = 250
    processed_qc[processed_qc == "Afectado por remanso"] = 250
    processed_qc[processed_qc == "Dato dudoso"] = 250
    processed_qc[processed_qc == 250] = 250
    
    # Create validation plot for this station if it has any estimated data
    estimated_mask = (df['Índice de calidad'] == "Dato estimado o calculado") & (processed_qc != 250)

    if estimated_mask.any():
        create_quality_validation_plot(df, estimated_mask, gauge_id)
    
    # Return mask of data to keep
    return (processed_qc == "Dato aceptado") | (processed_qc == "Dato estimado o calculado")


def create_quality_validation_plot(df, estimated_mask, gauge_id):
    """Create validation plot showing where short estimated periods were kept."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot streamflow
    ax1.plot(df.index, df['Valor'], label='Streamflow (m³/s)', alpha=0.7)
    
    # Highlight kept estimated periods
    estimated_data = df[estimated_mask]
    ax1.scatter(estimated_data.index, estimated_data['Valor'], 
               color='orange', alpha=0.5, label='Kept Estimated Data')

    # row = names_file.loc[names_file["gauge_name"] == gauge_id].iloc[0]
    # gauge_name = row["gauge_id"]

    # --- Fix: gauge_id is actually gauge_name here ---
    match = names_file.loc[names_file["gauge_name"] == gauge_id]
    if not match.empty:
        csv_gauge_id = match.iloc[0]["gauge_id"]
    else:
        csv_gauge_id = "Unknown ID"

    ax1.set_title(f'Gauge: {csv_gauge_id} ({gauge_id}) - Streamflow with Kept Estimated Periods')
    # ax1.set_title(f'Gauge {gauge_id}: ({gauge_name}) - Streamflow with Kept Estimated Periods')
    ax1.set_ylabel('Streamflow (m³/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot quality codes
    quality_colors = {
        "Dato aceptado": '#2ecc71',   # Good - Green
        # "Serie incompleta": '#f1c40f',   # Incomplete - Yellow
        "Dato estimado o calculado": '#e67e22',  # Estimated - Orange
        # "Augmented data": '#5DADE2', #Augmented data in light blue
        # 120: '#e74c3c',  # Suspect - Red
        # 200: '#95a5a6',  # Unchecked - Gray
        250: '#ffffff'   # Missing - White
    }

    # Plot quality bar
    ordered_codes = ["Dato aceptado", "Dato estimado o calculado", "Augmented data", 250]
    # for code in sorted(quality_colors.keys()):
    for code in ordered_codes:
        mask = df['Índice de calidad'] == code

        if code == "Augmented data":
            mask = (df['Índice de calidad'] == "Dato aceptado") & (df["Índice de revisión"] == "Augmented data")
            
        if mask.any():
            ax2.fill_between(df.index, 0, 1, where=mask, 
                           color=quality_colors[code], alpha=0.7)
    
    # Highlight kept estimated periods
    ax2.fill_between(df.index, 0, 1, where=estimated_mask, 
                    color='orange', alpha=0.3, label='Kept Estimated')
    
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Date')
    ax2.set_yticks([])
    
    # Add quality code legend
    quality_patches = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor="black", linewidth=0.6) 
                     for color in quality_colors.values()]
    quality_labels = ['Good', 'Estimated', 'Augmented', 'Missing']
    ax2.legend(quality_patches + [plt.Rectangle((0,0),1,1, facecolor='orange', edgecolor="black", linewidth=0.6, alpha=0.3)],
              quality_labels + ['Kept Estimated'],
              ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    
    plt.tight_layout()
    plt.savefig(validation_dir / f'quality_validation_{gauge_id}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

# Read all streamflow files
all_data = {}
processed_stations = []  # Keep track of all processed stations

# Also load strict high-quality data for comparison
strict_highqual_dir = Path("../processed_data/highqual")
try:
    strict_train = pd.read_csv(strict_highqual_dir / "train_data.csv", index_col=0, parse_dates=True)
    strict_val = pd.read_csv(strict_highqual_dir / "validation_data.csv", index_col=0, parse_dates=True)
    strict_test = pd.read_csv(strict_highqual_dir / "test_data.csv", index_col=0, parse_dates=True)
    strict_data = pd.concat([strict_train, strict_val, strict_test]).sort_index()
    has_strict_data = True
    print("\nLoaded strict high-quality data for comparison")
except FileNotFoundError:
    has_strict_data = False
    print("\nWarning: Could not find strict high-quality data for comparison")

# exclude_files = {"Paso de la Compania - caudal.xls", "Tacuarembo - caudal.xls", "Durazno - caudal.xls"}
exclude_files = {}

print("\nProcessing streamflow files...")
for file in path_to_xls_files.glob("*.xls"):
    if file.name in exclude_files:
        continue  # salta estos archivos
    # gauge_id = file.stem.split('_')[1]  # Extract gauge ID from filename
    gauge_id = file.stem.split(' - ')[0] #mine
    processed_stations.append(gauge_id)
    
    # Read the CSV file
    df = pd.read_excel(file)
    # Create date string and convert to datetime
    # df['date'] = df.apply(lambda x: f"{int(x['YYYY'])}-{int(x['MM']):02d}-{int(x['DD']):02d}", axis=1)
    # df['date'] = pd.to_datetime(df['date'])
    df['Fecha'] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y %H:%M")
    # Set date as index
    df = df.set_index('Fecha')

    # Fill missing dates with NaN to ensure continuous daily series
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_range)
    # Fill 'Indice de calidad' with 250 for newly created rows
    df['Índice de calidad'] = df['Índice de calidad'].fillna(250)
    
    # # First clean Dynkur data if this is gauge 1010
    # if gauge_id == '1010':
    #     df.loc[:, 'qobs'] = clean_dynkur_data(df['qobs'])
    
    # Then process quality codes with modified criteria
    high_quality_mask = process_quality_codes(df, gauge_id)
    
    # Set non-high-quality data to NaN
    df_processed = df.copy()
    df_processed.loc[~high_quality_mask, 'Valor'] = np.nan
    
    # # Clean specific periods for this gauge
    # df_processed = clean_specific_periods(df_processed, gauge_id)

    # --- Add curated streamflow values to original XLS ---
    # Ensure full daily coverage for both datasets
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    
    # Align original and processed data
    df_full = df.reindex(full_range)
    df_processed_full = df_processed.reindex(full_range)
    
    # Add curated streamflow column
    df_full['Valor_curado'] = df_processed_full['Valor']
    
    # Save to curated flows folder
    output_xls_path = curated_flows_dir / f"{gauge_id}_curated.xls"
    df_full.to_excel(output_xls_path, index_label='Fecha')
    
    print(f"Saved curated Excel for {gauge_id} → {output_xls_path.name}")
    
    all_data[gauge_id] = {
        'Valor': df_processed['Valor'],
        'Índice de calidad': df['Índice de calidad']
    }
    
#     # Create comparison plot if we have strict high-quality data
#     if has_strict_data and gauge_id in strict_data.columns:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
#         # Filter both datasets to the same period
#         start_date = '2000-10-01'
#         df_filtered = df_processed[df_processed.index >= start_date].copy()
#         strict_filtered = strict_data[strict_data.index >= start_date].copy()
        
#         # Plot both versions
#         ax1.plot(df_filtered.index, df_filtered['qobs'], 
#                 label='With Short Estimated', alpha=0.7)
#         ax1.plot(strict_filtered.index, strict_filtered[gauge_id], 
#                 label='Strict High-Quality', alpha=0.7)
        
#         # Calculate additional data points
#         common_index = df_filtered.index.intersection(strict_filtered.index)
#         df_aligned = df_filtered.reindex(common_index)
#         strict_aligned = strict_filtered[gauge_id].reindex(common_index)
        
#         strict_mask = ~strict_aligned.isna()
#         new_mask = ~df_aligned['qobs'].isna()
#         additional_mask = new_mask & ~strict_mask
        
#         if additional_mask.any():
#             ax1.scatter(common_index[additional_mask], 
#                        df_aligned['qobs'][additional_mask],
#                        color='orange', alpha=0.5, label='Additional Data Points')
        
#         ax1.set_title(f'Gauge {gauge_id} - Data Availability Comparison')
#         ax1.set_ylabel('Streamflow (m³/s)')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # Plot data availability comparison
#         availability = pd.DataFrame({
#             'Strict': ~strict_data[gauge_id].isna(),
#             'With Estimated': ~df_processed['qobs'].isna()
#         })
        
#         # Calculate monthly availability
#         monthly_avail = availability.resample('ME').mean() * 100
        
#         # Plot monthly availability
#         ax2.plot(monthly_avail.index, monthly_avail['Strict'], 
#                 label='Strict High-Quality', alpha=0.7)
#         ax2.plot(monthly_avail.index, monthly_avail['With Estimated'], 
#                 label='With Short Estimated', alpha=0.7)
        
#         # Plot difference where there's improvement
#         improvement = monthly_avail['With Estimated'] - monthly_avail['Strict']
#         improvement[improvement < 0] = 0  # Only show positive differences
#         ax2.bar(monthly_avail.index, improvement, 
#                 alpha=0.3, color='orange', label='Additional Availability')
        
#         # Format x-axis
#         ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
#         ax2.set_xlabel('Date')
#         ax2.set_ylabel('Monthly Data Availability (%)')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(validation_dir / f'comparison_{gauge_id}.png', 
#                     bbox_inches='tight', dpi=300)
#         plt.close()
        
#         # Print statistics using filtered data
#         strict_avail = (~strict_filtered[gauge_id].isna()).mean() * 100
#         new_avail = (~df_filtered['qobs'].isna()).mean() * 100
#         additional_points = additional_mask.sum()
        
#         print(f"\nComparison statistics for gauge {gauge_id}:")
#         print(f"Period: {start_date} onwards")
#         print(f"Strict high-quality availability: {strict_avail:.1f}%")
#         print(f"With short estimated availability: {new_avail:.1f}%")
#         print(f"Additional data points: {additional_points}")
#         print(f"Increase in availability: {new_avail - strict_avail:.1f}%")

#     # Dynkur data is already cleaned before quality processing

# # # Load catchment attributes and filter out human-influenced catchments
# # attrs_df = pd.read_csv(attrs_file_path, delimiter=';')
# # natural_catchments = attrs_df[attrs_df['degimpact'] != 's']['id'].astype(str).tolist()

# # # Remove downstream gauges 2, 32, 45 and 98, human influenced gauges 79, 96, 64, and gauge 6 (reservoir inflow estimation) and gauges 55 and 56 (watershed uncertain) and 83
# # downstream_gauges = ['2', '32', '45', '98', '96', '64', '79', '6','55', '56', '83']
# # natural_catchments = [x for x in natural_catchments if x not in downstream_gauges]

# # # Add Dynkur (1010) back to the list as we'll handle its influenced periods separately
# # if '1010' not in natural_catchments:
# #     natural_catchments.append('1010')

#DEBUG
print("Number of processed stations:", len(all_data))
for k, v in all_data.items():
    print(k, len(v['Valor']), type(v['Valor'].index))

# Combine all data into DataFrames
combined_df = pd.DataFrame({gauge_id: data['Valor'] for gauge_id, data in all_data.items()})

# DEBUG
print(type(combined_df.index), combined_df.index[:5])


quality_df = pd.DataFrame({gauge_id: data['Índice de calidad'] for gauge_id, data in all_data.items()})

# print("\n=== Station coverage check ===")
# for station in combined_df.columns:
#     print(station,
#           combined_df[station].index.min(),
#           combined_df[station].index.max(),
#           combined_df[station].notna().sum())


# Filter for the period we want (1999-10-01 onwards) and only natural catchments
# combined_df.index = pd.to_datetime(combined_df.index, errors='coerce') #mine
filtered_df = combined_df[combined_df.index >= '1980-01-01'].copy()
# filtered_df = filtered_df[[col for col in filtered_df.columns if col in natural_catchments]]

# print("\nProcessed stations:", sorted(processed_stations))
# print("Natural catchments:", sorted(natural_catchments))
# print("Stations after filtering:", sorted(filtered_df.columns.tolist()))

# Define training, validation, and test periods
train_period = pd.date_range(start='1999-10-01', end='2008-09-30', freq='D') #pd.date_range(start='1989-09-01', end='2001-08-31', freq='D')
val_period = pd.date_range(start='1989-10-01', end='1999-09-30', freq='D') #pd.date_range(start='2001-09-01', end='2005-08-31', freq='D')
test_period = pd.date_range(start='2008-10-01', end='2019-12-31', freq='D') #pd.date_range(start='2005-09-01', end='2009-08-31', freq='D')

# Split data into periods
train_data = filtered_df[filtered_df.index.isin(train_period)]
val_data = filtered_df[filtered_df.index.isin(val_period)]
test_data = filtered_df[filtered_df.index.isin(test_period)]

# print("\n=== Why stations are excluded ===")
# for station in filtered_df.columns:
#     data = filtered_df[station]
#     missing_ratio = data.isnull().mean()
#     train_avail = data.loc[data.index.intersection(train_period)].count()
#     val_avail   = data.loc[data.index.intersection(val_period)].count()
#     reasons = []
#     if missing_ratio > 0.5:
#         reasons.append(f"missing ratio too high ({missing_ratio:.2f})")
#     if train_avail < 365.25*6:
#         reasons.append(f"not enough train days ({train_avail})")
#     if val_avail < 365.25*2:
#         reasons.append(f"not enough val days ({val_avail})")
#     if reasons:
#         print(station, "excluded because:", "; ".join(reasons))
#     else:
#         print(station, "kept")


# Calculate statistics for each station
station_stats = {}
usable_stations = []

print("\nCalculating statistics for natural catchments only...")
for station in filtered_df.columns:  # Now filtered_df only contains natural catchments
    data = filtered_df[station]
    
    # Calculate overall missing ratio
    missing_ratio = data.isnull().mean()
    
    # Calculate yearly missing ratios
    yearly_missing = data.groupby(data.index.year).apply(lambda x: x.isnull().mean())
    bad_years = (yearly_missing > 0.50).sum()  # Count years with >50% missing data
    
    # # Calculate period statistics
    # train_data_station = data.loc[train_period]
    # val_data_station = data.loc[val_period]
    # test_data_station = data.loc[test_period]

    #calculate period statistics (mine)
    train_data_station = data.loc[data.index.intersection(train_period)]
    val_data_station   = data.loc[data.index.intersection(val_period)]
    test_data_station  = data.loc[data.index.intersection(test_period)]

    train_missing = train_data_station.isnull().mean()
    val_missing = val_data_station.isnull().mean()
    test_missing = test_data_station.isnull().mean()
    
    # Check if there's any data in each period
    has_train_data = not train_data_station.isnull().all()
    has_val_data = not val_data_station.isnull().all()
    has_test_data = not test_data_station.isnull().all()
    
    stats = {
        'total_days': len(data),
        'missing_days': data.isnull().sum(),
        'missing_percentage': missing_ratio * 100,
        'mean_flow': data.mean(),
        'min_flow': data.min(),
        'max_flow': data.max(),
        'bad_years': bad_years,
        'train_missing_ratio': train_missing,
        'val_missing_ratio': val_missing,
        'test_missing_ratio': test_missing,
        'has_train_data': has_train_data,
        'has_val_data': has_val_data,
        'has_test_data': has_test_data
    }
    station_stats[station] = stats
    
    # Calculate available days in each period
    train_available_days = train_data_station.count()  # Count non-NaN values
    val_available_days = val_data_station.count()
    
    # Required days (6 years in training, 2 years in validation)
    REQUIRED_TRAIN_DAYS = 365.25 * 6  # ~2191 days
    REQUIRED_VAL_DAYS = 365.25 * 2    # ~731 days
    
    # Apply filtering criteria:
    # 1. Overall missing ratio ≤ 50%
    # 2. Must have minimum required days in training and validation periods
    # 3. Test period data is optional
    if (missing_ratio <= 0.50 and 
        train_available_days >= REQUIRED_TRAIN_DAYS and 
        val_available_days >= REQUIRED_VAL_DAYS):
        usable_stations.append(station)

print(f"\nFound {len(usable_stations)} usable stations")
print("Usable stations:", sorted(usable_stations))

# Print data quality summary
print("\nData Quality Summary:")
print(f"Total catchments processed: {len(filtered_df.columns)}")
print(f"Usable catchments after filtering: {len(usable_stations)}")
print("\nPeriod Information:")
print(f"Training period: {train_period[0].strftime('%Y-%m-%d')} to {train_period[-1].strftime('%Y-%m-%d')}")
print(f"Validation period: {val_period[0].strftime('%Y-%m-%d')} to {val_period[-1].strftime('%Y-%m-%d')}")
print(f"Testing period: {test_period[0].strftime('%Y-%m-%d')} to {test_period[-1].strftime('%Y-%m-%d')}")

# Calculate and print average missing data percentages
print("\nAverage Missing Data Percentages for Usable Stations:")
print(f"Overall: {filtered_df[usable_stations].isnull().mean().mean()*100:.1f}%")
print(f"Training period: {train_data[usable_stations].isnull().mean().mean()*100:.1f}%")
print(f"Validation period: {val_data[usable_stations].isnull().mean().mean()*100:.1f}%")
print(f"Testing period: {test_data[usable_stations].isnull().mean().mean()*100:.1f}%")

# # Save the datasets
# train_data = filtered_df.loc[train_period, usable_stations]
# val_data = filtered_df.loc[val_period, usable_stations]
# test_data = filtered_df.loc[test_period, usable_stations]

# Save the datasets (mine)
train_data = filtered_df.reindex(train_period, columns=usable_stations)
val_data   = filtered_df.reindex(val_period, columns=usable_stations)
test_data  = filtered_df.reindex(test_period, columns=usable_stations)

train_data.to_csv(output_dir / "train_data.csv")
val_data.to_csv(output_dir / "validation_data.csv")
test_data.to_csv(output_dir / "test_data.csv")
combined_df.to_csv(output_dir / "all_data.csv")

# Save station statistics
stats_df = pd.DataFrame.from_dict(station_stats, orient='index')
stats_df.to_csv(output_dir / "station_statistics.csv")

# At the end, add overall comparison
if has_strict_data:
    print("\nOverall Comparison Statistics:")
    strict_stations = strict_train.columns
    modified_stations = train_data.columns
    common_stations = list(set(strict_stations) & set(modified_stations))
    
    print(f"\nNumber of usable stations:")
    print(f"Strict high-quality: {len(strict_stations)}")
    print(f"With short estimated: {len(modified_stations)}")
    
    if common_stations:
        # Filter both datasets to the same period
        start_date = '1980-09-01'
        strict_filtered = strict_data[strict_data.index >= start_date].copy()
        modified_filtered = filtered_df[filtered_df.index >= start_date].copy()
        
        strict_avail = strict_filtered[common_stations].notna().mean().mean() * 100
        modified_avail = modified_filtered[common_stations].notna().mean().mean() * 100
        
        print(f"\nAverage data availability for common stations:")
        print(f"Period: {start_date} onwards")
        print(f"Strict high-quality: {strict_avail:.1f}%")
        print(f"With short estimated: {modified_avail:.1f}%")
        print(f"Average increase: {modified_avail - strict_avail:.1f}%")
        
        # Create overall comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        strict_by_station = strict_filtered[common_stations].notna().mean() * 100
        modified_by_station = modified_filtered[common_stations].notna().mean() * 100
        
        ax1.scatter(strict_by_station, modified_by_station, alpha=0.6)
        ax1.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='1:1 Line')
        
        ax1.set_xlabel('Strict High-Quality Availability (%)')
        ax1.set_ylabel('With Short Estimated Availability (%)')
        ax1.set_title('Data Availability Comparison by Station')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Distribution plot
        improvement = modified_by_station - strict_by_station
        bins = np.linspace(0, improvement.max(), 20)
        ax2.hist(improvement, bins=bins, alpha=0.7)
        ax2.axvline(improvement.mean(), color='r', linestyle='--', 
                   label=f'Mean: {improvement.mean():.1f}%')
        
        ax2.set_xlabel('Increase in Data Availability (%)')
        ax2.set_ylabel('Number of Stations')
        ax2.set_title('Distribution of Availability Improvement')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(validation_dir / 'overall_comparison.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()

print("\nProcessing complete!")
print(f"Results saved in: {output_dir}")
print("Check validation_dir for comparison plots and statistics")
