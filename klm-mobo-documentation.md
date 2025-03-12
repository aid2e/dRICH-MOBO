# KLM MOBO Usage Instructions

This repo uses multi-objective Bayesian optimization (MOBO) to optimize the detector geometry for the EIC KLM. The framework uses [Ax](https://ax.dev/) and [BoTorch](https://botorch.org/) for machine learning, and [EIC software](https://eic.phy.anl.gov/tutorials/eic_tutorial/) to run detector simulations. Consult the relevant documentation for more information.

**Optimization pipeline:**
- Set optimization information in `optimize.config` and `parameters.config`
- Information is read by `wrapper_slurm_basic.py`, which constructs a search space and generation strategy for the specified objectives
- Scheduler first runs a set number of quasi-random trials, then optimized trials
    - `ProjectUtils/ePICUtils/editxml.py` updates geometry to reflect current trial's parameters
    - Trials run as slurm jobs
    - Each slurm job runs `ddsim` particle simulation runs with `ProjectUtils/ePICUtils/shell_wrapper_job.sh`
    - `ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py` calculates objectives from simulation results
    - Objectives are used to update model before deploying new optimized trials
- Final results exported in `test_scheduler_df.csv`, `test_scheduler_experiment.json`, and `gs_model.pkl` 

## Initial installation
Clone the `klm` branch of the `dRICH-MOBO` repo with:
```bash
git clone -b klm https://github.com/aid2e/dRICH-MOBO.git
```
Run the setup script to set environment variables and install `eic-shell` and the `one_sector_solenoid` branch of [`epic_klm`](https://github.com/simons27/epic_klm/tree/one_sector_solenoid) into the `MOBO-tools` directory.
```bash
cd dRICH-MOBO/MOBO-tools
source setup.sh
```
Build ePIC geometry in `epic_klm` through `eic-shell`
```bash
./eic-shell
cd epic_klm
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=install
cmake --build build
cmake --install build
exit
```
Create a virtualenv with the required packages using `uv`
```bash
uv sync
```

## Optimization setup

**Parameters**
- Define each parameter in `parameters.config`
    - `"lower"` and `"upper"` refer to the bounds the parameter can take in the optimization search space
    - `"default"` refers to the nominal value in the detector
    - `"units"` refers to the unit of the parameter; set it to an empty string if unitless
    - `"path"` refers to the path to the parameter's value in `epic_klm/compact/pid/klmws.xml`
- Make sure parameters are correctly updated in xml file by `ProjectUtils/ePICUtils/editxml.py`
- Make sure parameters are correctly set to default values in `status_quo_arm` in `wrapper_slurm_basic.py`

**Objectives**
- Set metric names in `ProjectUtils/slurm_utilities.py`
- In `wrapper_slurm_basic.py`, set:
    - metric names in `names`
    - type of metric (float or int) in `RangeParameter`
    - whether to minimize or maximize metric in `SlurmJobMetric`
    - objective thresholds in `ObjectiveThreshold`
        - (The objective threshold should be set to slightly lower than your minimum acceptable value for a maximized metric, and vice versa for a minimized metric)
    - default metric values in `status_quo_metric_vals`
        - (These values are the calculated objectives corresponding to the default parameters in `parameters.config`)
- Set metric noise in `sem` in `ProjectUtils/metric_utilities.py`
    - For noiseless observations, set `sem = 0.0`
    - To have Ax infer the noise level, set `sem = None`
    - *Note: Ax does not allow hardcoding the noise level for some objectives and inferring for others*


**Calculations**
- Calculate each objective metric in `ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py`
    - The current implementation runs `ProjectUtils/ePICUtils/genMomentumScan_job.sh` and `ProjectUtils/ePICUtils/shell_wrapper_job.sh` as part of calculating metrics; this may change for different implementations

**Optimization configuration**
- Set optimization information in `optimize.config`
    - `n_calls` refers to the number of trials run by the optimization (not counting the initial nominal/status quo trial)
- Set model generation information in `GenerationStrategy` in `wrapper_slurm_basic.py`
    - `model`:
        - `Models.SOBOL` generates initial quasi-random points to explore the search space
        - `Models.BOTORCH_MODULAR` uses `qNoisyExpectedHypervolumeImprovement` to exploit better outcomes
    - `num_trials` refers to the number of trials to run for each generation step (`-1` means to run until `n_calls` is reached)
    - `max_parallelism` refers to how many trials can be run in parallel at once
- Set slurm job parameters in `submit_slurm_job` in `ProjectUtils/slurm_utilities.py` and `makeSlurmScript` in `ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py`

## Running optimization
Start the trials with
```bash
source run_optimization.sh
```
This script will activate the conda environment, set environment variables, check that `eic-shell` and `epic_klm` are installed, then run `wrapper_slurm_basic.py`.

While running, current slurm jobs can be viewed through [DCC OnDemand](https://dcc-ondemand-01.oit.duke.edu/pun/sys/dashboard/activejobs) or with `squeue -u <netID>`

## Output analysis
- `test_scheduler_df.csv` contains a table of information for each trial
- `test_scheduler_experiment.json` contains information for the Ax experiment
- `gs_model.pkl` contains the model created by the generation strategy

**Loading results**
- Import necessary modules:
```python
from wrapper_slurm_basic import *
from ax.storage.json_store.load import load_experiment
```
- Load experiment and model files:
```python
bundle = RegistryBundle(metric_clss={SlurmJobMetric: None}, runner_clss={SlurmJobRunner: None})
experiment = load_experiment(filepath='test_scheduler_experiment.json', decoder_registry=bundle.decoder_registry)
with open('gs_model.pkl', 'rb') as file:
    model = pickle.load(file)
```

**Interpreting results**
- Create [plots](https://ax.readthedocs.io/en/latest/plot.html)

- Load results into a dataframe with:

```python
experiment.fetch_data().df
```


- Create a dictionary of parameter and objective metric values for each trial:

```python
data_df = experiment.fetch_data().df
metric_num = len(experiment.metrics)

mobo_data = {
    data_df.loc[i * metric_num].arm_name :
        {
            'parameters' :
                experiment.arms_by_name[data_df.loc[i * metric_num].arm_name].parameters,
            'metrics' :
                {
                    data_df.loc[i * metric_num + j]['metric_name'] :
                        data_df.loc[i * metric_num + j]['mean']
                    for j in range(metric_num)
                }
        }
    for i in range(experiment.num_trials)
}
```