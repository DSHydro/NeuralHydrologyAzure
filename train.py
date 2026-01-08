from pathlib import Path
import torch
from neuralhydrology.nh_run import start_run, eval_run
from cloudpathlib import CloudPath

config = CloudPath("https://raw.githubusercontent.com/neuralhydrology/neuralhydrology/refs/heads/master/examples/01-Introduction/1_basin.yml")

# Train
# -----------------------------
# by default we assume that you have at least one CUDA-capable NVIDIA GPU or MacOS with Metal support
if torch.cuda.is_available() or torch.backends.mps.is_available():
    start_run(config_file=config)
# fall back to CPU-only mode
else:
    start_run(config_file=config, gpu=-1)

# Evaluate Test Set
# -----------------------------
# TODO: change path
run_dir = Path("runs/test_run_0501_214945")
eval_run(run_dir=run_dir, period="test")


