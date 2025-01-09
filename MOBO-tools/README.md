# Python dependencies

Dependencies from '../conda_requirements.txt' or '../pip_requirements.txt' should be installed (ported from Closure-test 1).

# Required EIC software 
The detector simulation utilizes the full ePIC software stack. The EIC singularity container wrapper ```eic-shell``` will need to be installed to compile the multi-mirror dRICH code and run simulations (installation instructions at https://eicrecon.epic-eic.org/#/get-started/eic-shell). This repository is set up to use forks from tagged versions of ePIC and EICrecon, meaning that the compatible version of the singularity container must also be used. If you dont have access to previous tagged versions of the container via CVFMS, one can be installed by calling the install script (https://github.com/eic/eic-shell/blob/main/install.sh) with the desired version.

**Current required eic-shell version:** 'eic_xl 24.11.1-stable'

# Simulation setup

Before building ePIC software, call `source setup.sh` to set some needed environment variables.

The following repositories/branches hold the current geometry used for the optimization of the dRICH and will need to be downloaded and build prior to running:
- **EICrecon (fork):** ``` git clone -b v1.19-drich-2mirror https://github.com/cpecar/EICrecon-drich-mobo.git ```
- **ePIC (fork):** ``` git clone -b 24.11.1-drich-2mirror https://github.com/cpecar/epic-geom-drich-mobo.git ```
- **IRT:** ``` git clone -b multi-mirror-irt https://github.com/eic/irt.git ```

These forks contain tagged versions of the ePIC geometry and reconstruction, with changes to the dRICH geometry and reconstruction procedures to support multiple spherical mirrors with different radii.

These can each be built using the `build_*.sh` scripts from **within eic-shell**, and the build location should be set to $EIC_SOFTWARE (required for ensuring EICrecon loads the correct irt version),

``` ./build_epic.sh epic-geom-drich-mobo $EIC_SOFTWARE ```

``` ./build_eicrecon.sh EICrecon-drich-mobo $EIC_SOFTWARE ```

``` ./build_irt.sh irt $EIC_SOFTWARE ```

Additionally, the dRICH analysis script will need to be built from within eic-shell prior to running the optimization, which can be done via `cd ProjectUtils/ePICUtils/ && make`.

# Running MOBO wrapper

The current MOBO wrapper script can be run using 
```python wrapper_slurm_basic.py -c optimize.config -d parameters.config```

Currently, this by default utilizes a slurm trial scheduler which will submit each trial as a slurm job, with each simulation point carried out in additional slurm jobs (configured for the Duke compute cluster, but partitions etc. can be edited to reflect any other cluster in ProjectUtils/slurm_utilities.py and ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py). joblib and PanDA/iDDS wrappers under development.
