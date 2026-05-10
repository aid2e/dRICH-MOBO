# Plan: Replace Layer-Based muID with GNN-Based Classification

## Overview

Replace the current mu/pi ID approach (ROCAUC.py, layer-based pixel counting at fixed momenta) with a GNN-based binary classifier trained on the same data pipeline as the neutron energy objectives. The classifier uses the full momentum range (0.5-5 GeV), and AUC is binned at 2.75 GeV to produce low_muID_auc and high_muID_auc.

## Design Decisions

- **New script**: `train_GNN_classifier.py` (standalone, does not modify `train_GNN.py`)
- **Data pipeline**: Reuse `submit_workflow.py` with a new `--skipTraining` flag for mu- and pi+ data production
- **Model**: Full GIN with graph convolution + sigmoid output + BCELoss
- **AUC binning**: 2.75 GeV energy threshold (same as RMSE split)
- **Sim settings**: 80 sims x 500 events per particle (same as neutrons)
- **Momentum range**: Same as neutrons (steering file controls range, ~0.5-5 GeV)

---

## Step 1: Modify `submit_workflow.py`

**File**: `/hpc/group/vossenlab/rck32/eic/work_eic/slurm/submit_workflow.py`

**Changes**:
- Add `--skipTraining` argument (BooleanOptionalAction, default False)
- When `--skipTraining` is set, skip the training job submission block entirely (skip `submit_training_job()` call and its monitoring loop)
- Everything else stays the same: sim+process+analyze jobs still run, CSVs are still produced

**Why**: This lets us reuse the full DDSim -> process_root_file.py -> analyze_data.py pipeline for mu- and pi+ without running the energy regression training.

---

## Step 2: Create `train_GNN_classifier.py`

**File**: `/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/train_GNN_classifier.py`

**Based on**: GNN_PID.ipynb workflow

**Script responsibilities**:
1. Load CSVs from two particles (mu- and pi+) using different input prefixes
   - mu- CSVs: `{inputDataPrefMu}{i}.csv` for i in range(numDfs)
   - pi+ CSVs: `{inputDataPrefPi}{j}.csv` for j in range(numDfs)
   - Assign different `file_idx` ranges to avoid collision (mu: 0..N-1, pi: N..2N-1)
2. Apply `process_df_vectorized(data, cone_angle_deg=40)` for cone filtering
3. Create `HitDataset(data, filter_events=True, connection_mode="kNN", k=6, function='PID')`
   - Uses `PDG_label_dict` to map truePID to binary labels (muon=1, pion=0)
4. Stratified train/val/test split (70/15/15) balancing classes
5. Initialize full GIN model with sigmoid output (from GNN_PID.ipynb)
   - hidden_dim=32, n_conv_layers=2, n_linear_layers=7, linear_capacity=5, lr=1e-4
   - BCELoss criterion
6. Train with early stopping (limit=4)
7. Evaluate on test set: compute binned ROC AUC
   - For each test event, compute energy = sqrt(mass^2 + P^2)
   - Events with energy < 2.75 GeV -> low energy bin
   - Events with energy >= 2.75 GeV -> high energy bin
   - Compute `roc_auc_score()` separately for each bin
8. **Append** results to the results file (mode "a"):
   ```
   \n{low_muID_auc}
   \n{high_muID_auc}
   ```

**Command-line arguments**:
- `--inputDataPrefMu` (str): CSV prefix for muon data
- `--inputDataPrefPi` (str): CSV prefix for pion data
- `--numDfs` (int): Number of CSV files per particle
- `--resultsFilePath` (str): Path to append AUC results
- `--modelPath` (str): Directory to save trained model
- `--runName` (str): Run name for logging
- `--deleteDfs` (bool): Whether to delete CSVs after training

**Imports from GNN_util.py**: `process_df_vectorized`, `HitDataset`, `create_directory`

**GIN model definition**: Copy the GIN class from GNN_PID.ipynb (with sigmoid) directly into the script (or import from GNN_util if it's there). The key difference from the regression GIN: final layer outputs 1 value + sigmoid activation.

**train_GNN function**: Copy from GNN_PID.ipynb (the version with BCELoss, accuracy tracking, and early stopping).

**test/eval function**: New `compute_binned_auc()` function that:
- Iterates through test_dataloader
- For each event: gets prediction probability, true label, and energy
- Bins by 2.75 GeV threshold
- Returns (low_auc, high_auc)

---

## Step 3: Modify `newRunTestsAndObjectiveCalc.py`

**File**: `/hpc/group/vossenlab/rck32/eic/dRICH-MOBO/MOBO-tools/ProjectUtils/ePICUtils/newRunTestsAndObjectiveCalc.py`

### 3a: New method `makeSlurmScript_data_production(self, particle)`

Generates a SLURM script that calls `submit_workflow.py --skipTraining` for a given particle. Similar to existing `makeSlurmScript()` but:
- Passes `--particle {particle}` (e.g., "mu-" or "pi+")
- Passes `--skipTraining` flag
- Uses a particle-specific run_name prefix: `mobo_{self.job_id}_{particle_safe}` (where particle_safe replaces +/- with p/m for filename safety)
- Does NOT pass `--outFile` (no results file needed from this step)
- Does NOT pass energy objective flags (not applicable)

### 3b: New method `createClassificationTrainingJob(self)`

Generates a SLURM script that:
- Uses GPU partition (`scavenger-gpu`)
- Activates ML venv
- Runs `train_GNN_classifier.py` with:
  - `--inputDataPrefMu {workdir}/macros/Timing_estimation/data/df/mobo_{jobid}_mu-_500events_run_{jobid}_`
  - `--inputDataPrefPi {workdir}/macros/Timing_estimation/data/df/mobo_{jobid}_pip_500events_run_{jobid}_`
  - `--numDfs 80`
  - `--resultsFilePath {self.outname}` (the klm-mobo-out file)
  - `--modelPath {model_dir}`

### 3c: New method `runClassificationTrainingJob(self)`

Submits the classification training SLURM job and stores its ID for monitoring.

### 3d: New method `monitorClassificationJob(self)`

Same pattern as `monitorROCAUCJob()` — polls until complete.

### 3e: Update `runJobs()`

**Old flow**:
```python
if run_mu_pi_objectives:
    for p_point in p_points:
        makeSlurmScript_mupi(p_point)  # DDSim at fixed momentum
if run_neutron_objectives:
    makeSlurmScript()  # full neutron pipeline
```

**New flow**:
```python
if run_mu_pi_objectives:
    makeSlurmScript_data_production("mu-")   # sim+process+analyze for muons
    makeSlurmScript_data_production("pi+")   # sim+process+analyze for pions
if run_neutron_objectives:
    makeSlurmScript()  # full neutron pipeline (unchanged)
```

All three data production jobs run in parallel.

### 3f: Update post-monitoring logic

**Old**:
```python
if run_mu_pi_objectives:
    runJobs_ROCAUC()
    monitorROCAUCJob()
```

**New**:
```python
if run_mu_pi_objectives:
    runClassificationTrainingJob()
    monitorClassificationJob()
```

### 3g: Update `p_scan` usage

The `p_scan` variable and fixed-momentum logic are no longer needed for mu/pi objectives. The mu/pi data now uses the same momentum range as neutrons (controlled by steering file). The `p_scan` variable can remain but is only used if the old ROCAUC approach is ever re-enabled.

### 3h: Update `run_root_files=False` fallback

Line 338: `manager.final_job_status = [1,1]` needs to account for 3 jobs now (mu- data, pi+ data, neutron). Change to `[1,1,1]` or dynamically set based on which objectives are active.

### 3i: Clean up file deletion

The old root file deletion loop (lines 349-354) deletes `scan_{jobid}_{particle}_p_{p}.edm4hep.root`. This is no longer relevant since the new pipeline uses submit_workflow.py which handles its own file cleanup via `--deleteROOTFile` and `--deleteJSON` flags. Remove or update this section.

---

## Step 4: No changes needed to these files

- **`wrapper_slurm_basic.py`** — Already updated with 4 objectives (user's recent edits)
- **`slurm_utilities.py`** — Already updated with 4 objective names
- **`optimize.config`** — Already updated (n_objectives=4, n_design_params=3)
- **`parameters.config`** — Already updated with basic params (pending the fixes noted earlier)
- **`editxml.py`** — Already handles basic params correctly
- **`GNN_util.py`** — HitDataset already supports `function='PID'`, GIN class exists. May need to verify the GIN class has sigmoid variant, otherwise define it in train_GNN_classifier.py directly.

---

## Results File Assembly (Critical Ordering)

The `klm-mobo-out_{jobid}.txt` file is built in two stages:

1. **train_GNN.py** (neutron energy) writes with mode **"w"**:
   ```
   {low_RMSE}
   {high_RMSE}
   ```

2. **train_GNN_classifier.py** (mu/pi PID) appends with mode **"a"**:
   ```
   \n{low_muID_auc}
   \n{high_muID_auc}
   ```

This matches the objectives order in `slurm_utilities.py`: `["low_RMSE", "high_RMSE", "low_muID_auc", "high_muID_auc"]`

**CRITICAL**: The neutron job (which writes with "w") MUST complete before the classification job (which appends with "a") starts. This is already guaranteed by the flow: `retrieveResults()` reads the RMSE file first, then `runClassificationTrainingJob()` is called after.

---

## Execution Order Summary

```
1. Overlap check
2. Submit 3 parallel data production jobs:
   a. Neutron: submit_workflow.py (sim + process + analyze + train) → writes RMSE to outfile
   b. Muon:    submit_workflow.py --skipTraining (sim + process + analyze only) → CSVs
   c. Pion:    submit_workflow.py --skipTraining (sim + process + analyze only) → CSVs
3. Monitor all 3 jobs until complete
4. retrieveResults() — verify RMSE values from outfile
5. Submit classification training job (train_GNN_classifier.py on GPU)
6. Monitor classification job until complete
7. Results file now has all 4 objective values
```

---

## Edge Cases / Gotchas

1. **Run name collision**: The mu- and pi+ run names must be distinct to avoid CSV filename collisions. Use particle name in the prefix: `mobo_{jobid}_mu-_` vs `mobo_{jobid}_pip_`.

2. **file_idx in HitDataset**: When loading CSVs from two particles, the `file_idx` column must be unique across both. The classifier script assigns mu: 0..N-1, pi: N..2N-1 (matching GNN_PID.ipynb pattern).

3. **Particle name in filenames**: `pi+` has a `+` which can be problematic in filenames/shell args. submit_workflow.py already handles this since it passes `--gun.particle pi+` to DDSim. But for file paths, verify the CSV naming works correctly.

4. **GPU availability**: The classification training job needs a GPU (scavenger-gpu partition). The neutron training job already uses this. Both shouldn't be submitted simultaneously to the same GPU — but since classification runs AFTER neutron completes, this is fine.

5. **Energy calculation for binning**: In the classifier's eval, energy = sqrt(mass^2 + P^2). The mass depends on truePID: muon mass = 0.10566 GeV, pion mass = 0.13957 GeV. These are already in HitDataset's mass_dict.

6. **submit_workflow.py environment variables**: It requires WORK_EIC, EIC_SHELL_HOME, ML_VENV_HOME, MAIL_USER, EPIC_HOME. The MOBO setup.sh sets WORK_EIC and EPIC_HOME but may not set ML_VENV_HOME and MAIL_USER. Verify these are set in the work_eic/setup.sh that gets sourced.
