import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pickle
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import yaml

from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config

# ------- Paths -------
model_dir=Path('../CHIRPS_2.0/runs/gauge_and_chirps_precip_seq_270_30_epochs_seq_270_hidden_256_dropout_04_fb_05_seed111_0402_180903')
pickle_file_dir = model_dir / 'validation' / 'model_epoch030' / 'validation_results.p'
pt_file_dir = model_dir / 'model_epoch030.pt'
optimizer_file_dir = model_dir / 'optimizer_state_epoch030.pt'
scaler_path = model_dir / "train_data" / "train_data_scaler.yml"

pt_data = torch.load(pt_file_dir,weights_only=True)

cfg = Config(model_dir / "config.yml")

model = get_model(cfg)

model.load_state_dict(torch.load(pt_file_dir))
model.eval()

with open(scaler_path, "r") as f:
    scaler = yaml.safe_load(f)

# LOADING ATTRIBUTES
attributes_file = Path("../CHIRPS_2.0/filtered_data_gauge_and_CHIRPS/attributes/attributes.csv")

df_attr = pd.read_csv(attributes_file, index_col=0)

# --- Define the features your model uses ---
static_features = [
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "p_mean",
    "pet_mean",
    "aridity",
    "high_prec_dur",
    "low_prec_dur"
]

# Ensure the index is a string
df_attr.index = df_attr.index.astype(str)

# Select your basin
basin = "CAMELS_UY_6"

# static_values
static_values = df_attr.loc[basin, static_features].to_numpy(dtype=np.float32)

attr_means = np.array([scaler["attribute_means"][k] for k in static_features], dtype=np.float32)
attr_stds  = np.array([scaler["attribute_stds"][k]  for k in static_features], dtype=np.float32)

static_norm = (static_values - attr_means) / attr_stds
static_tensor = torch.tensor(static_norm).unsqueeze(0)  # [1, n_static]

# DYNAMIC INPUTS

def norm_dyn(tensor, varname):
    center = scaler["xarray_feature_center"]["data_vars"][varname]["data"]
    scale  = scaler["xarray_feature_scale"]["data_vars"][varname]["data"]
    center = torch.tensor(center, dtype=tensor.dtype, device=tensor.device)
    scale  = torch.tensor(scale,  dtype=tensor.dtype, device=tensor.device)
    return (tensor - center) / scale


ts_file = Path("../CHIRPS_2.0/filtered_data_gauge_and_CHIRPS/time_series") / f"{basin}.nc"

ds = xr.open_dataset(ts_file)

dynamic_vars = ["QObs_mm_d","prcp_mm_day","srad_W_m2","tmax_C","tmin_C","prcp_chirps_mm_day"]

# --- Convert xarray to DataFrame and sort by date ---
df_dyn = ds[dynamic_vars].to_dataframe().sort_index()

t0_date = pd.to_datetime("2008-10-01")

seq_length = 270  # same as your model’s training

# Find the index of t0
t0_idx = df_dyn.index.get_loc(t0_date)

# Slice the dynamic data for the sequence before t0
start_idx = t0_idx - seq_length + 1  # inclusive
history_df = df_dyn.iloc[start_idx : t0_idx + 1][dynamic_vars]

# --- Convert to tensor ---
historical_dynamic_tensor = torch.tensor(history_df.values.astype(np.float32)).unsqueeze(0)

# historical_dynamic_tensor: [1, 270, 6] as you already computed
x_prcp      = historical_dynamic_tensor[..., 1:2]
x_srad      = historical_dynamic_tensor[..., 2:3]
x_tmax      = historical_dynamic_tensor[..., 3:4]
x_tmin      = historical_dynamic_tensor[..., 4:5]
x_prcp_ch   = historical_dynamic_tensor[..., 5:6]

x_prcp_norm    = norm_dyn(x_prcp,    "prcp_mm_day")
x_srad_norm    = norm_dyn(x_srad,    "srad_W_m2")
x_tmax_norm    = norm_dyn(x_tmax,    "tmax_C")
x_tmin_norm    = norm_dyn(x_tmin,    "tmin_C")
x_prcp_ch_norm = norm_dyn(x_prcp_ch, "prcp_chirps_mm_day")

history_inputs = {
    "x_d": {
        "prcp_mm_day":        x_prcp_norm,
        "srad_W_m2":          x_srad_norm,
        "tmax_C":             x_tmax_norm,
        "tmin_C":             x_tmin_norm,
        "prcp_chirps_mm_day": x_prcp_ch_norm,
    },
    "x_s": static_tensor
}

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
model = get_model(cfg).to(device)
state_dict = torch.load(pt_file_dir, map_location=device)
model.load_state_dict(state_dict)
model.eval()

history_inputs = {
    "x_d": {k: v.to(device) for k, v in history_inputs["x_d"].items()},
    "x_s": history_inputs["x_s"].to(device)
}

with torch.no_grad():
    out = model(history_inputs)

h0 = out["h_n"]
c0 = out["c_n"]
y_hat = out["y_hat"]

q_center = scaler["xarray_feature_center"]["data_vars"]["QObs_mm_d"]["data"]
q_scale  = scaler["xarray_feature_scale"]["data_vars"]["QObs_mm_d"]["data"]

q_center = torch.tensor(q_center, dtype=y_hat.dtype, device=y_hat.device)
q_scale  = torch.tensor(q_scale,  dtype=y_hat.dtype, device=y_hat.device)

y_hat_denorm = y_hat * q_scale + q_center