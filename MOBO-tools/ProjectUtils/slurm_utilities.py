import os, sys, subprocess
import numpy as np
from ax.core.base_trial import TrialStatus
from time import time
from ProjectUtils.ePICUtils.editxml import create_xml
import pandas as pd

from typing import Any, Dict, NamedTuple, Union
import os
def check_and_create_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
class SlurmJob(NamedTuple):
    id: int
    slurmid: int
    parameters: Dict[str, Union[str, float, int, bool]]
    
class SlurmQueueClient:
    """ Class to queue and query slurm jobs,
    based on https://ax.dev/tutorials/scheduler.html
    """
    jobs = {}
    totaljobs = 0
    objectives = [
              "low_RMSE",
             "high_RMSE",
             "sepMuPi_1GeV",
             "sepMuPi_5GeV"#,
#              "outer_radius"
                  ]
    
    def submit_slurm_job(self, jobnum):
        jobdir = str(os.environ["AIDE_HOME"])+ f"/OUTDIR/job_{jobnum}/"
        check_and_create_directory(jobdir)
        with open("{}jobconfig_{}.slurm".format(jobdir,jobnum),"w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=klm-mobo\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=10:00:00\n") #CHECK HOW LONG IS REALLY NEEDED
            file.write(f"#SBATCH --output={jobdir}klm-mobo_%j.out\n")
            file.write(f"#SBATCH --error={jobdir}klm-mobo_%j.err\n")
            
            file.write("python " + str(os.environ["AIDE_HOME"])+"/ProjectUtils/ePICUtils/"+"newRunTestsAndObjectiveCalc.py {} \n".format(jobnum))

        shellcommand = ["sbatch","{}jobconfig_{}.slurm".format(jobdir,jobnum)]        
        commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
        
        output = commandout.stdout.decode('utf-8')
        line_split = output.split()
        if len(line_split) == 4:
            return int(line_split[3])
        else:
            return -1
        return
    
    def schedule_job_with_parameters(self, parameters):
        ### HERE: schedule the slurm job, retrieve the jobid from command line output        
        ### totaljobs/jobid defines the suffix of the xml files we will use
        create_xml(parameters, self.totaljobs)
        
        slurmjobnum = self.submit_slurm_job(self.totaljobs)
        jobid = self.totaljobs
        
        self.jobs[jobid] = SlurmJob(jobid, slurmjobnum, parameters)        
        self.totaljobs += 1        
        return jobid
    
    def get_job_status(self, jobid):
        job = self.jobs[jobid]
        
        if job.slurmid == -1:
            # something failed in job submission
            return TrialStatus.FAILED

        ### HERE: run bash command to retrieve status, exit code
        shellcommand = [str(os.environ["AIDE_HOME"])+"/ProjectUtils/"+"checkSlurmStatus.sh", str(job.slurmid)]
        commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
        
        output = commandout.stdout.decode('utf-8')
        line_split = output.split()

        if len(line_split) == 1:
            status = line_split[0]
        else:
            #something wrong, try again
            print("Error in checking slurm status, assuming still running")
            return TrialStatus.RUNNING

        if status == "0":
            return TrialStatus.RUNNING
        elif status == "1":
            return TrialStatus.COMPLETED
        elif status == "-1":
            return TrialStatus.FAILED
        
        return TrialStatus.RUNNING

    def get_outcome_value_for_completed_job(self, jobid):
        job = self.jobs[jobid]
        ### HERE: load results from text file, formatted based on job id
        results = np.loadtxt(os.environ["AIDE_HOME"]+"/log/results/" + "klm-mobo-out_{}.txt".format(jobid))
        if len(self.objectives) > 1:            
            results_dict = {self.objectives[i]:results[i] for i in range(len(self.objectives))}
        else:
            results_dict = {self.objectives[0]:results}
        return results_dict

SLURM_QUEUE_CLIENT = SlurmQueueClient()

def get_slurm_queue_client():
    return SLURM_QUEUE_CLIENT
