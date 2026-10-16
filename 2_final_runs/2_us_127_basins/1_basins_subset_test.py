from pathlib import Path
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import xarray as xr
import yaml
import geopandas as gpd
import contextily as cx
from adjustText import adjust_text
import itertools


from neuralhydrology.nh_run import start_run, eval_run, finetune
from neuralhydrology.nh_run import continue_run
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester, metrics

# -------- Paths ---------
CONFIG_PATH = Path("./1_basins_subset_test.yml")
RUNS_DIR = Path("runs")
ATTRIBUTES_FILE= Path('../../1_preparing_data/2_us_127_basins/data/attributes/attributes_other.csv')

seeds = [111, 222, 333, 444, 555, 666, 777, 888] #  

base_non_precip_inputs = [
    "temperature_2m_max",
    "temperature_2m_min",
    "surface_net_solar_radiation_mean",
]

with open(CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()

# for r in range(1, len(precip_products) + 1):

precip_combos = [
    # ("camels_precipitation",),
    # ("total_precipitation_sum",),
    # ("chirps_precipitation",),
    # ("mswep_precipitation",),
    # ("chirps_precipitation", "mswep_precipitation"),
    # ("total_precipitation_sum", "chirps_precipitation"),
    # ("total_precipitation_sum", "mswep_precipitation"),
    ("mswep_precipitation", "camels_precipitation"),
    # ("total_precipitation_sum", "chirps_precipitation", "mswep_precipitation"),
]

# for precip_combo in itertools.combinations(precip_products, r):
for precip_combo in precip_combos:
    for seed in seeds:
        config = base_config.copy()

        config["dynamic_inputs"] = [*base_non_precip_inputs, *precip_combo]
        config["seed"] = seed

        precip_name = "_".join(precip_combo)
        config["experiment_name"] = (
            f"{precip_name}_seq_{config['seq_length']}"
            f"_{config['predict_last_n']}_epochs_{config['epochs']}"
            f"_hidden_{config['hidden_size']}"
            f"_dropout_{str(config['output_dropout']).replace('.', '')}"
            f"_fb_{config['initial_forget_bias']}"
            f"_seed{seed}"
        )

        temp_config_path = Path(f"temp_{precip_name}_seed{seed}.yml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        print(f"Running: {config['experiment_name']}")

        if use_gpu:
            start_run(config_file=temp_config_path)
        else:
            start_run(config_file=temp_config_path, gpu=-1)

        temp_config_path.unlink()  # clean up temp file after run