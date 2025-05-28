# Python dependencies
- Ax version 0.3.6 and compatible version of BoTorch (both installed via ```pip install ax-platform==0.3.6```)
- matplotlib
- wandb

# Required EIC software 
The detector simulation utilizes the full ePIC software stack. The EIC singularity container wrapper ```eic-shell``` will need to be installed to compile the multi-mirror dRICH code and run simulations (installation instructions at https://eicrecon.epic-eic.org/#/get-started/eic-shell). This repository is set up to use forks from tagged versions of ePIC and EICrecon, meaning that the compatible version of the singularity container must also be used. If you dont have access to previous tagged versions of the container via CVFMS, one can be installed by calling the install script (https://github.com/eic/eic-shell/blob/main/install.sh) with the desired version.

**Current required eic-shell version:** 'eic_xl 24.11.1-stable'

# Environment variables
Before building ePIC software, call `source setup.sh` to set some needed environment variables. In this script are variables which can be changed to reflect the location of the `eic-shell` script (if not in MOBO-tools), the output directory for simulation files, and the build location for the ePIC geometry. 

# EIC simulation software
Some changes to the dRICH geometry were necessary to directly set some design parameters in the optimization procedure, avoiding some overlaps between sensors and the walls of the dRICH, and for studies such as optimizing a dRICH with multiple mirrors with unique radii. For this reason, running the dRICH optimization requires cloning and building a fork of the ePIC geometry description.

It is recommended to use the branch of the fork containing a single-mirror dRICH, which is then compatible with the already built versions of `EICrecon` and `IRT` available in `eic-shell:24.11.1-stable`.

The following repository and branch hold the recommended default geometry for the optimization of the dRICH and will need to be downloaded and built prior to running:
- **ePIC (fork):** ``` git clone -b 24.11.1-drich-singlemirror https://github.com/cpecar/epic-geom-drich-mobo.git ```

This can each be built using the `build_epic.sh` script from **within eic-shell**, and the build location should be set to $EIC_SOFTWARE,

``` ./build_epic.sh epic-geom-drich-mobo $EIC_SOFTWARE ```

Additionally, the dRICH analysis script will need to be built from within eic-shell prior to running the optimization, which can be done via `cd ProjectUtils/ePICUtils/ && make`.

After building the ePIC fork, call `source load_epic.sh` to set necessary environment variables to direct to the local installation.

# Running MOBO wrapper

The current MOBO wrapper script can be run using 
```python wrapper_slurm_basic.py -c optimize.config -d parameters.config```

Currently, this by default utilizes a custom slurm Runner in the Ax scheduler which will submit each trial as a slurm job, with each simulation point carried out in additional slurm jobs (configured for the Duke compute cluster, but partitions etc. can be edited to reflect any other cluster in ProjectUtils/slurm_utilities.py and ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py). joblib and PanDA/iDDS wrappers under development.
