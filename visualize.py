from pathlib import Path
import matplotlib.pyplot as plt
from neuralhydrology.evaluation import metrics
import pickle
import sys

run_dir = Path(sys.argv[1])
output = run_dir / "test" / "model_epoch003" / "test_timeseries.png"

with open(run_dir / "test" / "model_epoch003" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

results.keys()

# extract observations and simulations
DS = results['01022500']['1D']['xr']
qobs = DS['QObs(mm/d)_obs']
qsim = DS['QObs(mm/d)_sim']

fig, ax = plt.subplots(figsize=(16,10))
ax.plot(qobs['date'], qobs)
ax.plot(qsim['date'], qsim)
ax.set_ylabel("Discharge (mm/d)")
ax.set_title(f"Test period - NSE {results['01022500']['1D']['NSE']:.3f}")
plt.savefig(output)
print(f"Saved plot to {output}")

values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
for key, val in values.items():
    print(f"{key}: {val:.3f}")
