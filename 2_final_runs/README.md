# 2_final_runs

This folder contains the final training and testing experiments for the NeuralHydrology models. Each subfolder corresponds to a different basin set and experimental configuration.

## Subfolders

### `1_531_caravan_vs_camels`
Training and testing of models on 531 CARAVAN basins, comparing the effect of using CARAVAN vs. CAMELS precipitation inputs.

### `2_us_127_basins`
Training and testing of models on a subset of 127 US CAMELS basins, evaluating model performance across multiple precipitation products.

### `3_uy_basins`
Training and testing of models on the Uruguayan CAMELS-UY basins, including hydrograph visualization and peak flow evaluation.

## Structure

Each subfolder follows the same general structure:

- **Notebooks** — step-by-step workflows for basin selection, model training, testing, and ensemble metric computation
- **`.yml` config files** — NeuralHydrology run configurations
- **`runs/`** — model run outputs generated during training
- **`ensemble_metrics/` / `ensemble_testing_metrics/` / `ensemble_peak_metrics/`** — aggregated evaluation results across runs
- **`data/`** — basin input data (if present, they are copied from `1_preparing_data`)
