**Setup instructions** 

Dependencies from ../conda_requirements.txt or ../pip_requirements.txt should be installed (ported from Closure-test 1).

```source setup.sh``` will set needed environment variables and check if [eic/epic](https://github.com/eic/epic/tree/main) and 'eic-shell' are installed in this directory. If not, they will be cloned/downloaded.

**First time installing**

If this is your first time installing this repo:
- Follow build instructions in 'epic' to build the ePIC geometry within eic-shell (will only need built once, as geometry editing currently only requires changing the xml files)

**Running MOBO wrapper**

The current MOBO wrapper script can be run using 
```python wrapper_slurm_basic.py -c optimize.config -d parameters.config```

Currently, this by default utilizes a slurm trial scheduler which will submit each trial as a slurm job, with each simulation point carried out in additional slurm jobs (configured for the Duke compute cluster, but partitions etc. can be edited to reflect any other cluster in ProjectUtils/slurm_utilities.py and ProjectUtils/ePICUtils/runTestsAndObjectiveCalc.py). joblib and PanDA/iDDS wrappers will be under development.

```wrapper.py``` is currently out of date with other pieces of the workflow, such as editing the dRICH geometry, and will fail if used.