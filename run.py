from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch
import time
import pickle
import yaml

from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester, metrics
from neuralhydrology.utils.errors import AllNaNError

# --------------------------- Paths ---------------------------
PROJECT_ROOT = Path(".")  # we already %cd'ed into the project root
CONFIG_PATH  = PROJECT_ROOT / "config.yml"
DATA_DIR     = PROJECT_ROOT / "data"
TS_DIR       = (DATA_DIR / "time_series") if (DATA_DIR / "time_series").exists() else (DATA_DIR / "forcings")
ATTR_DIR     = DATA_DIR / "attributes"
RUNS_DIR     = PROJECT_ROOT / "runs"
PLOTS_DIR    = PROJECT_ROOT / "evaluation_plots"

# --------------------------- Helpers ---------------------------
def _ensure_dirs():
    for p in [DATA_DIR, TS_DIR, ATTR_DIR, RUNS_DIR, PLOTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def _first_nc():
    nc_files = sorted(TS_DIR.glob("*.nc"))
    if not nc_files:
        print(f"[ERROR] No NetCDF files found in {TS_DIR.resolve()}.")
        print("        Put your per-basin .nc files there (variables must include your dynamic_inputs and target).")
        sys.exit(1)
    return nc_files[0]

def _validate_vars_against_netcdf(cfg: Config):
    """Check that target_variables and dynamic_inputs exist in the data; warn or fail gracefully."""
    nc = _first_nc()
    try:
        ds = xr.open_dataset(nc)
    except Exception as e:
        print(f"[ERROR] Failed to open {nc}: {e}")
        sys.exit(1)

    data_vars = set(ds.data_vars)
    ds.close()

    targets = list(getattr(cfg, "target_variables", []))
    dynamics = list(getattr(cfg, "dynamic_inputs", []))

    # Targets must exist (hard requirement)
    missing_targets = [v for v in targets if v not in data_vars]
    if missing_targets:
        print(f"[ERROR] target_variables missing in NetCDF: {missing_targets}")
        print(f"        Found variables in sample file {nc.name}: {sorted(data_vars)[:20]} ...")
        sys.exit(1)

    # Warn and drop any dynamics that aren't present (soft requirement)
    missing_dyn = [v for v in dynamics if v not in data_vars]
    if missing_dyn:
        print(f"[WARN] dynamic_inputs not found in NetCDF and will be dropped: {missing_dyn}")
        dynamics = [v for v in dynamics if v in data_vars]
        if not dynamics:
            print("[ERROR] After dropping missing dynamics, no dynamic_inputs remain.")
            sys.exit(1)
        cfg.update_config({"dynamic_inputs": dynamics})

    # Also ensure we didn't accidentally include the target as a dynamic input
    dyn_overlap = [v for v in dynamics if v in targets]
    if dyn_overlap:
        print(f"[INFO] Removing target from dynamic_inputs to avoid leakage: {dyn_overlap}")
        cfg.update_config({"dynamic_inputs": [v for v in dynamics if v not in targets]})

def _summarize_config(cfg: Config):
    print("\n=== EFFECTIVE CONFIG (key bits) ===")
    print("run_dir:", getattr(cfg, "run_dir", None))
    print("data_dir:", getattr(cfg, "data_dir", None))
    for k in ["train_basin_file","validation_basin_file","test_basin_file"]:
        print(f"{k}:", getattr(cfg, k, None))
    print("target_variables:", getattr(cfg, "target_variables", None))
    print("dynamic_inputs  :", getattr(cfg, "dynamic_inputs", None))
    print("static_attributes:", getattr(cfg, "static_attributes", None))
    print("device:", getattr(cfg, "device", None))
    print("===================================\n")

def _dump_cfg_no_overwrite(cfg: Config, base_dir: Path, prefer_name: str = "config_patched.yml") -> Path:
    """Write cfg to a new filename to avoid overwriting the original config.yml."""
    out = base_dir / prefer_name
    if out.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = base_dir / f"{out.stem}_{ts}{out.suffix}"
    cfg.dump_config(folder=out.parent, filename=out.name)
    return out

# --------------------------- Train / Evaluate ---------------------------
def train_model():
    if not CONFIG_PATH.exists():
        print(f"[ERROR] Missing config at {CONFIG_PATH}.")
        sys.exit(1)

    _ensure_dirs()

    cfg = Config(CONFIG_PATH)
    # # Make sure paths in the config match your actual files
    # _fix_basin_files_in_config(cfg)
    # # Validate variables against your NetCDFs; drop missing dynamics, avoid leaking target
    # _validate_vars_against_netcdf(cfg)

    # # Persist any updates we made to a *new* config file (no overwrite)
    # patched_cfg_path = _dump_cfg_no_overwrite(cfg, PROJECT_ROOT)
    # print(f"[INFO] Wrote patched config to {patched_cfg_path}")

    _summarize_config(cfg)

    print("Starting training …")
    use_gpu = torch.cuda.is_available() and str(getattr(cfg, "device", "cuda:0")).startswith("cuda")
    if use_gpu:
        print("→ Training on GPU")
        # start_run(config_file=patched_cfg_path)
        start_run(config_file=CONFIG_PATH)
    else:
        print("→ Training on CPU")
        # start_run(config_file=patched_cfg_path, gpu=-1)
        start_run(config_file=CONFIG_PATH, gpu=-1)
    print("Training complete.")

def evaluate_and_plot():
    if not RUNS_DIR.exists():
        print(f"[ERROR] No 'runs' directory at {RUNS_DIR.resolve()}.")
        sys.exit(1)

    try:
        # latest_run = sorted(RUNS_DIR.glob("*"))[-1] #this orders alphabetically
        latest_run = max(RUNS_DIR.glob("*"), key=lambda d: d.stat().st_mtime)
    except IndexError:
        print(f"[ERROR] No runs found in {RUNS_DIR.resolve()}.")
        sys.exit(1)

    run_config_path = latest_run / "config.yml"
    if not run_config_path.exists():
        print(f"[ERROR] Missing run config at {run_config_path}.")
        sys.exit(1)

    cfg = Config(run_config_path)
    print(f"\nEvaluating latest run: {latest_run.name}")

    # Prefer 'test' period; fall back to 'validation' if needed
    period = "test"
    try:
        tester = get_tester(cfg=cfg, run_dir=latest_run, period=period, init_model=True)
    except Exception as e:
        print(f"[WARN] Could not init tester for period '{period}': {e}")
        period = "validation"
        tester = get_tester(cfg=cfg, run_dir=latest_run, period=period, init_model=True)

    results = tester.evaluate(save_results=True, metrics=cfg.metrics)

    PLOTS_DIR.mkdir(exist_ok=True)

    def _pick_freq_key(by_freq: dict):
        for k in ["1D","D","daily"]:
            if k in by_freq:
                return k
        return next(iter(by_freq.keys()))

    def _extract_series(xr_block, cfg):
        # infer obs/sim variable names
        tgt = None
        if hasattr(cfg, "target_variables") and cfg.target_variables:
            tgt = cfg.target_variables[0]
        keys = list(xr_block.data_vars)
        obs_key = next((k for k in keys if k.endswith("_obs")), tgt or "qobs")
        sim_key = next((k for k in keys if k.endswith("_sim")), f"{obs_key}_sim")
        return xr_block[obs_key], xr_block[sim_key]

    print("\nCreating evaluation plots…")
    missing_counter = 0

    for basin_id, by_freq in results.items():
        try:
            fkey = _pick_freq_key(by_freq)
            xr_block = by_freq[fkey]["xr"]
            qobs, qsim = _extract_series(xr_block, cfg)

            # pick time coordinate if present
            time_coord = next((c for c in ["time","date","time_step"] if c in qobs.coords), None)
            time = qobs[time_coord].values if time_coord else range(qobs.shape[0])

            # collapse extra dims if needed (e.g., ensemble/lead)
            for name in ["qobs", "qsim"]:
                arr = locals()[name]
                
                # for dim in list(arr.dims):
                #     if dim not in ["time","date"] and arr.sizes.get(dim, 1) > 1:
                #         locals()[name] = arr.isel({dim: -1})
        
            # qobs_arr, qsim_arr = locals()["qobs"], locals()["qsim"]

                for dim in list(arr.dims):
                    if dim not in ["time", "date"]:
                        # collapse singleton dims or take last element for larger dims
                        arr = arr.isel({dim: 0}) if arr.sizes[dim] == 1 else arr.isel({dim: -1})
                
                # assign back to the proper local variable
                if name == "qobs":
                    qobs_arr = arr
                else:
                    qsim_arr = arr
            
            nse_val = metrics.calculate_metrics(qobs_arr, qsim_arr, metrics=["NSE"])["NSE"]

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(time, qobs_arr.values, label="Observed", alpha=0.7)
            ax.plot(time, qsim_arr.values, label="Simulated", alpha=0.7)
            ax.set_title(f"{period.title()} — Basin {basin_id} — NSE: {nse_val:.3f}")
            ax.set_xlabel("Date"); ax.set_ylabel("Streamflow (mm/d)")    #(cfg.target_variables[0] if hasattr(cfg, "target_variables") else "Streamflow (mm/d)")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = PLOTS_DIR / latest_run.name / f"basin_{basin_id}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            missing_counter += 1
            print(f"[WARN] Skipping basin {basin_id}: {e}")

    print(f"Evaluation plots saved to: {out_path.resolve()}")
    if missing_counter:
        print(f"[NOTE] Skipped {missing_counter} basin(s) due to missing/unexpected result structure.")

# --------------------------- After traininig ---------------------------
def negative_outputs():
    if not RUNS_DIR.exists():
        print(f"[ERROR] No 'runs' directory at {RUNS_DIR.resolve()}.")
        sys.exit(1)

    try:
        # latest_run = sorted(RUNS_DIR.glob("*"))[-1] #this orders alphabetically
        latest_run = max(RUNS_DIR.glob("*"), key=lambda d: d.stat().st_mtime)
    except IndexError:
        print(f"[ERROR] No runs found in {RUNS_DIR.resolve()}.")
        sys.exit(1)

    run_path = latest_run
    
    # Load run configuration
    config_path = run_path / "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract info for labeling
    batch_size = config.get("batch_size", None)
    epochs = config.get("epochs", None)
    learning_rate = config.get("learning_rate", None)
    hidden_size = config.get("hidden_size", None)
    seq_length = config.get("seq_length", None)
    output_dropout = config.get("output_dropout", None)

    # Find all model files
    model_files = list(run_path.glob("model_epoch*.pt"))
    if not model_files:
        print(f"No model files found in {run_path}")
        return
    
    # Pick the last model file
    last_model_file = max(model_files, key=lambda f: int(f.stem.split("model_epoch")[-1]))
    last_model_name = last_model_file.stem

    # Load test results
    results_path = run_path / "test" / f"{last_model_name}" / "test_results.p"
    with open(results_path, "rb") as fp:
        results = pickle.load(fp)

    def _pick_freq_key(by_freq: dict):
        for k in ["1D","D","daily"]:
            if k in by_freq:
                return k
        return next(iter(by_freq.keys()))

    def _extract_series(xr_block, config):
        tgt = None
        if hasattr(config, "target_variables") and config.target_variables:
            tgt = config.target_variables[0]
        keys = list(xr_block.data_vars)
        obs_key = next((k for k in keys if k.endswith("_obs")), tgt or "qobs")
        sim_key = next((k for k in keys if k.endswith("_sim")), f"{obs_key}_sim")
        return xr_block[obs_key], xr_block[sim_key]

    # Prepare output file
    output_dir = Path(PROJECT_ROOT) / "negative_values"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{latest_run.name}.txt"

    print(f"Looking for negative values of simulated streamflow in run '{latest_run.name}'")
    basins_with_negatives = 0  # counter
    
    with open(output_file, "w") as f:
        f.write('-------------Configuration summary-------------\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Hidden size: {hidden_size}\n')
        f.write(f'Sequence length: {seq_length}\n')
        f.write(f'Dropout: {output_dropout}\n')
        f.write('-----------------------------------------------\n')
        for basin_id, by_freq in results.items():
            try:
                fkey = _pick_freq_key(by_freq)
                xr_block = by_freq[fkey]["xr"]
                qobs, qsim = _extract_series(xr_block, config)

                time_coord = next((c for c in ["time","date","time_step"] if c in qobs.coords), None)
                time = qobs[time_coord].values if time_coord else range(qobs.shape[0])

                for name in ["qobs", "qsim"]:
                    arr = locals()[name]
                    for dim in list(arr.dims):
                        if dim not in ["time", "date"]:
                            arr = arr.isel({dim: 0}) if arr.sizes[dim] == 1 else arr.isel({dim: -1})
                    if name == "qobs":
                        qobs_arr = arr
                    else:
                        qsim_arr = arr
                        
                basin_has_negative = False
                f.write(f'-------------------Basin: {basin_id}-------------------\n')
                for i in range(len(qsim_arr)):
                    if qsim_arr[i] < 0:
                        f.write(f'Date: {time[i]}. Value: {qsim_arr[i].item():.2f} mm/d.\n')
                        basin_has_negative = True

                if basin_has_negative:
                    basins_with_negatives += 1

            except Exception as e:
                f.write(f"[WARN] Could not process basin {basin_id}: {e}\n")

        # Summary line
        total_basins = len(results)
        f.write('\n=================== Summary ===================\n')
        f.write(f'Basins with at least one negative prediction: {basins_with_negatives} out of {total_basins}\n')
        f.write('===============================================\n')
    
    print(f"Negative values written to {output_file}")

def plot_metrics():
    """
    Load results from a single run directory, calculate metrics for each basin,
    and create/save boxplots for all metrics grouped by basin_number.
    """

    if not RUNS_DIR.exists():
        print(f"[ERROR] No 'runs' directory at {RUNS_DIR.resolve()}.")
        sys.exit(1)

    try:
        # latest_run = sorted(RUNS_DIR.glob("*"))[-1] #this orders alphabetically
        latest_run = max(RUNS_DIR.glob("*"), key=lambda d: d.stat().st_mtime)
    except IndexError:
        print(f"[ERROR] No runs found in {RUNS_DIR.resolve()}.")
        sys.exit(1)

    run_path = latest_run

    # Create output directory for plots
    output_dir = Path(f"../metric_plots/{run_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []

    # Load run configuration
    config_path = run_path / "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract info for labeling
    batch_size = config.get("batch_size", None)
    epochs = config.get("epochs", None)
    learning_rate = config.get("learning_rate", None)
    hidden_size = config.get("hidden_size", None)
    seq_length = config.get("seq_length", None)
    initial_forget_bias = config.get("initial_forget_bias", None)

    # Build a short run-info string for labeling
    run_info = (
        f"bs={batch_size}, ep={epochs}, lr={learning_rate},\n"
        f"hs={hidden_size}, sl={seq_length}, fb={initial_forget_bias}"
    )

    # Find all model files
    model_files = list(run_path.glob("model_epoch*.pt"))
    if not model_files:
        print(f"No model files found in {run_path}")
        return
    
    # Pick the last model file
    last_model_file = max(model_files, key=lambda f: int(f.stem.split("model_epoch")[-1]))
    last_model_name = last_model_file.stem

    # Load test results
    results_path = run_path / "test" / f"{last_model_name}" / "test_results.p"
    with open(results_path, "rb") as fp:
        results = pickle.load(fp)

    # Collect metrics across all basins
    for _, result in results.items():
        qobs = result['1D']['xr']['streamflow_obs']
        qsim = result['1D']['xr']['streamflow_sim']
    
        try:
            metric_dict = metrics.calculate_all_metrics(
                qobs.isel(time_step=-1),
                qsim.isel(time_step=-1)
            )
        except AllNaNError:
            continue
    
        # Store metrics for this basin
        row = dict(metric_dict) if isinstance(metric_dict, dict) else dict(metric_dict.to_dict())
        data.append(row)

    if not data:
        print("No valid metrics to plot.")
        return

    df = pd.DataFrame(data)
    metric_names = list(df.columns)

    # --- Plotting: one boxplot per metric (values across basins) ---
    for metric in metric_names:
        values = df[metric].dropna().values

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.boxplot(values, labels=[metric],showfliers=False)
        # ax.set_xlabel(f"{run_info}")
        ax.set_ylabel(metric)
        ax.set_title(f"{run_info}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png")
        plt.close(fig)

    print(f"Saved metric plots to {output_dir}")

# --------------------------- Run ---------------------------
# Basic presence checks
if not CONFIG_PATH.exists():
    print(f"[ERROR] {CONFIG_PATH} not found. Put your config.yml at the project root.")
    raise SystemExit(1)
if not DATA_DIR.exists():
    print(f"[ERROR] {DATA_DIR} not found. Create it and add 'time_series' (or 'forcings') and 'attributes'.")
    raise SystemExit(1)
if not TS_DIR.exists():
    print(f"[ERROR] {TS_DIR} not found. Create it and add your NetCDF files.")
    raise SystemExit(1)

t0 = time.time()
train_model()
evaluate_and_plot()
# %run viz_losses.py
negative_outputs()
plot_metrics()
print(f"Training took {(time.time()-t0)/60:.2f} minutes")