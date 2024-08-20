** Python dependencies ** 

Dependencies from ../conda_requirements.txt or ../pip_requirements.txt should be installed (ported from Closure-test 1).

** Required EIC software **
The detector simulation utilizes the full ePIC software stack. The EIC singularity container ```eic-shell``` will need to be installed to compile the multi-mirror dRICH code and run simulations (installation instructions at https://eicrecon.epic-eic.org/#/get-started/eic-shell).

Before building ePIC software, call ```source setup.sh``` to set some needed environment variables.

The following repositories/branches hold the current geometry used for the optimization of the dRICH and will need to be downloaded and build prior to running:
- EICrecon (fork): ``` git clone -b v1.15-drich-3mirror https://github.com/cpecar/EICrecon-drich-mobo.git ```
- ePIC (fork): ``` git clone -b 24.07.0-drich-3mirror https://github.com/cpecar/epic-geom-drich-mobo.git ```
- IRT: ``` git clone -b multi-mirror-irt https://github.com/eic/irt.git ```

These can each be built using the ```build_*.sh``` scripts, and the build location should be set to $EIC_SOFTWARE (required for ensuring EICrecon loads the correct irt version).

**Running MOBO wrapper**

The current MOBO wrapper script can be run using 
```python wrapper_slurm_basic.py -c optimize.config -d parameters.config```

Currently, this by default utilizes a slurm trial scheduler which will submit each trial as a slurm job, with each simulation point carried out in additional slurm jobs (configured for the Duke compute cluster, but partitions etc. can be edited to reflect any other cluster in ProjectUtils/slurm_utilities.py and ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py). joblib and PanDA/iDDS wrappers under development.

```wrapper.py``` is currently out of date with other pieces of the workflow, such as editing the dRICH geometry, and will fail if used.