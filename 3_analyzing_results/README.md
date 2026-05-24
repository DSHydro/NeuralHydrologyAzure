# 3_analyzing_results

This folder contains the post-processing and analysis of model outputs.
## Subfolders

### `1_531_caravan_vs_camels`
Analysis of model performance across 531 CARAVAN basins, comparing results when using CARAVAN vs. CAMELS precipitation inputs. Also contains the notebook used to select the final 127 US basin subset based on NSE score distributions and differences between precipitation products.

### `2_us_127_basins`
Analysis of model runs for the 127 US CAMELS basin subset, including precipitation product characterization, aggregated run metrics, and statistical difference testing across products.

### `3_uy_basins`
Analysis of model runs for the Uruguayan CAMELS-UY basins, following the same structure as `2_us_127_basins` with precipitation characterization, run metrics, and statistical comparisons.

### `4_precip_analysis`
Cross-basin precipitation analysis focusing on peak flow metrics, summary plots, and statistical differences across precipitation products.
