from pathlib import Path
import matplotlib.pyplot as plt
from neuralhydrology.evaluation import metrics
import pickle
import sys

run_dir = Path(sys.argv[1])
# output = run_dir / "test" / "model_epoch003" / "test_timeseries.png"

with open(run_dir / "test" / "model_epoch003" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

for key in results.keys():
    # extract observations and simulations
    DS = results[key]['1D']['xr']
    print(DS)
    qobs = DS['streamflow_obs']
    qsim = DS['streamflow_sim']
    
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(qobs['date'], qobs, label='Observed')
    ax.plot(qsim['date'], qsim, label='Simulated')
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_title(f"Test period - NSE {results[key]['1D']['NSE']:.3f}")
    ax.legend()
    fig.tight_layout()

    output = run_dir / "test" / "model_epoch003" / f"{key}_test_timeseries.png"
    plt.savefig(output)
    plt.close(fig)
    print(f"Saved plot to {output}")

# values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
# for key, val in values.items():
#     print(f"{key}: {val:.3f}")
