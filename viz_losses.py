import os, glob, pandas as pd, matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import yaml

# Find latest run
PROJECT_ROOT = ".."
run = sorted(glob.glob(os.path.join(PROJECT_ROOT, "runs", "*")), key=os.path.getmtime)[-1]
run_name = os.path.basename(run)
print(f"\nFound latest run: {run}")

# Create output directory
eval_dir = Path("../evaluation")
run_dir = eval_dir / run_name
loss_dir = run_dir / "loss"
loss_dir.mkdir(parents=True, exist_ok=True)
print(f"Created directory: {loss_dir}")


# Load config.yml to get loss function   <-- NEW BLOCK
config_path = Path(run) / "config.yml"
loss_name = None
if config_path.exists():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        loss_name = config.get("loss", None)
        print(f"Found loss function in config.yml: {loss_name}")
else:
    print("No config.yml found in run directory")
    

# Load tensorboard data
print("\nLoading tensorboard data...")
ea = EventAccumulator(run)
ea.Reload()
scalar_tags = ea.Tags().get("scalars", [])
print(f"Found {len(scalar_tags)} scalar tags:")
for tag in scalar_tags:
    print(f"  - {tag}")

def as_df(tag):
    ev = ea.Scalars(tag)
    df = pd.DataFrame({"step":[e.step for e in ev], "value":[e.value for e in ev]})
    print(f"Loaded {len(df)} values for {tag}")
    return df

# Heuristics: plot anything that looks like train/val losses or metrics
wanted = [t for t in scalar_tags
          if any(k in t.lower() for k in ["loss","train","val","valid","nse","kge","rmse"])]
print(f"\nFound {len(wanted)} relevant tags:")
for tag in wanted:
    print(f"  - {tag}")

print(f"\nSaving plots to: {loss_dir}")

# Store DataFrames for train and validation losses
train_loss_df = None
val_loss_df = None

for tag in wanted:
    df = as_df(tag)
    if df.empty: 
        print(f"Skipping {tag} - no data")
        continue
        
    # Store loss DataFrames for combined plot
    if "train/avg_loss" in tag:  # Changed from "train" and "loss" to exact match
        train_loss_df = df
        print(f"Found training loss data: {len(df)} points")
    if "valid/avg_loss" in tag:  # Changed from "val" or "valid" and "loss" to exact match
        val_loss_df = df
        print(f"Found validation loss data: {len(df)} points")
        
    # Create individual plot
    plt.figure(figsize=(12, 6))
    ax = df.plot(x="step", y="value", grid=True, legend=False)

    
    title = tag
    if "loss" in tag.lower() and loss_name:
        title += f" ({loss_name})"
    ax.set_title(title)

    
    # ax.set_title(tag)
    ax.set_xlabel("step/epoch")
    ax.set_ylabel("value")
    
    # Save plot
    safe_tag = "".join(c if c.isalnum() else "_" for c in tag)  # Make filename safe
    out_path = loss_dir / f"{safe_tag}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

# Create combined loss plot if we have both train and validation losses
if train_loss_df is not None and val_loss_df is not None:
    print("\nCreating combined loss plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.plot(train_loss_df['step'], train_loss_df['value'], 
             'b-', label='Training Loss', alpha=0.7)
    
    # Plot validation loss
    plt.plot(val_loss_df['step'], val_loss_df['value'], 
             'r-', label='Validation Loss', alpha=0.7)

    # ---- NEW: append loss function to title if available ----
    title = "Training and Validation Loss"
    if loss_name:
        title += f" ({loss_name})"
    plt.title(title)
    # ---------------------------------------------------------
    
    # plt.title('Training and Validation Loss')
    plt.xlabel('step/epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = loss_dir / "combined_train_val_loss.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {out_path}")
else:
    print("\nCouldn't create combined plot - missing train or validation loss data")

print("\nDone!")
