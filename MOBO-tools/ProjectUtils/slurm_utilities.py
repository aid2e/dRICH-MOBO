import os, sys, subprocess
import numpy as np
from ax.core.base_trial import TrialStatus
from time import time
from ProjectUtils.ePICUtils.editxml import create_xml

from typing import Any, Dict, NamedTuple, Union

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
    metrics = None
    #["piKsep_etalow",
    #              "piKsep_etahigh",
    #              "acceptance"
    #              ]
    
    def submit_slurm_job(self, jobnum):
        with open("jobconfig_{}.slurm".format(jobnum),"w") as file:
            # currently configured for Slurm on the Duke Compute Cluster
            # TODO: set these parameters from a config filfe
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=drich-mobo\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=scavenger\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=4:00:00\n") 
            file.write("#SBATCH --output=/hpc/group/vossenlab/cmp115/AIDE/dRICH-MOBO/MOBO-tools/log/job_output/drich-mobo_%j.out\n")
            file.write("#SBATCH --error=/hpc/group/vossenlab/cmp115/AIDE/dRICH-MOBO/MOBO-tools/log/job_output/drich-mobo_%j.err\n")
            
            file.write("python " + str(os.environ["AIDE_HOME"])+"/ProjectUtils/ePICUtils/"+"/runTestsAndObjectiveCalc.py {} \n".format(jobnum))
        print("starting slurm job ", jobnum)
        shellcommand = ["sbatch","jobconfig_{}.slurm".format(jobnum)]        
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
            #something wrong, assume still running and check again
            print("Error in checking slurm status, assuming still running")
            return TrialStatus.RUNNING
        
        if status == "0":
            return TrialStatus.RUNNING
        elif status == "1":
            return TrialStatus.COMPLETED
        elif status == "-2":
            return TrialStatus.EARLY_STOPPED
        elif status == "-1":
            return TrialStatus.FAILED
        
        return TrialStatus.RUNNING
        
    def get_outcome_value_for_completed_job(self, jobid):
        job = self.jobs[jobid]
        ### HERE: load results from text file, formatted based on job id
        results = np.loadtxt(os.environ["AIDE_HOME"]+"/log/results/" + "drich-mobo-out_{}.txt".format(jobid))
        results_dict = {}
        if len(self.metrics) > 1:
            for idx, obj in enumerate(self.metrics):
                mean = results[2*idx]
                sem = results[2*idx+1]
                if sem == 0:
                    sem = None
                results_dict[obj] = [mean,sem]
        else:
            results_dict = {self.metrics[0]:results}
        return results_dict

SLURM_QUEUE_CLIENT = SlurmQueueClient()

def get_slurm_queue_client():
    return SLURM_QUEUE_CLIENT
