# Imports
from pathlib import Path
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import xarray as xr

from neuralhydrology.nh_run import start_run, eval_run, finetune
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester, metrics


# -------- Paths ---------
CONFIG_PATH = Path("fine-tuning/entire_camels.yml")
RUNS_DIR = Path("runs")
PLOTS_DIR = Path("evaluation_plots")

# by default we assume that you have at least one CUDA-capable NVIDIA GPU or MacOS with Metal support
if torch.cuda.is_available() or torch.backends.mps.is_available():
    start_run(config_file=CONFIG_PATH)

# fall back to CPU-only mode
else:
    start_run(config_file=CONFIG_PATH, gpu=-1)

# === Identify Most Recent Run Directory ===
if not RUNS_DIR.exists():
    sys.exit(f"[ERROR] No 'runs' directory found at {RUNS_DIR.resolve()}.")

try:
    run_dir = max(RUNS_DIR.glob("*/"), key=lambda d: d.stat().st_mtime)
except ValueError:
    sys.exit(f"[ERROR] No run folders found in {RUNS_DIR.resolve()}.")

print(f"\nUsing latest run directory: {run_dir.name}")


# === Detect the Last Epoch Folder ===
val_dir = run_dir / "validation"
if not val_dir.exists():
    sys.exit(f"[ERROR] No validation directory found in {run_dir}.")

# Find all epoch folders like model_epoch001, model_epoch002, ...
epoch_dirs = [d for d in val_dir.glob("model_epoch*") if d.is_dir()]
if not epoch_dirs:
    sys.exit(f"[ERROR] No epoch folders found in {val_dir}.")

# Get the one with the highest epoch number
last_epoch_dir = sorted(epoch_dirs, key=lambda p: int(p.name.split("epoch")[-1]))[-1]
print(f"Using last epoch folder: {last_epoch_dir.name}")

val_metrics_path = last_epoch_dir / "validation_metrics.csv"
if not val_metrics_path.exists():
    sys.exit(f"[ERROR] Validation metrics file not found: {val_metrics_path}")


# === Load Validation Results ===
df = pd.read_csv(val_metrics_path, dtype={'basin': str}).set_index('basin')
median_nse = df['NSE'].median()
print(f"\nMedian NSE (validation period): {median_nse:.3f}")

# Select a random basins from the lower 50% of the NSE distribution
basin = df.loc[df["NSE"] < df["NSE"].median()].sample(n=1).index[0]

print(f"Selected basin: {basin} with an NSE of {df.loc[df.index == basin, 'NSE'].values[0]:.3f}")

# -------- Fine-tuning --------
print("\n=== Starting Fine-tuning ===")
# !cat finetune.yml

finetune(Path("fine-tuning/finetune.yml"))

# Automatically detect newest run (fine-tuned)
finetune_dir = max(RUNS_DIR.glob("*/"), key=lambda d: d.stat().st_mtime)
print(f"\nEvaluating fine-tuned model from: {finetune_dir.name}")
eval_run(finetune_dir, period="test")

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

evaluate_and_plot()
