import numpy as np
import os, sys, subprocess, time
import math

#NEEDS TO:
# 1. run overlap check
# 2. generate p/eta scan
# 3. calculate pi-K sep
# 4. store all relevant results in drich-mobo-out_{jobid}.txt

class SubJobManager:
    def __init__(self, p_eta_points, n_part, job_id):
        self.p_eta_points = p_eta_points
        self.n_part = n_part
        self.job_id = job_id
        self.outname = str(os.environ["AIDE_HOME"])+"/log/results/"+ "drich-mobo-out_{}.txt".format(jobid)
        self.slurm_job_ids = []
        
    def checkOverlap(self):
        shellcommand = [str(os.environ["EPIC_MOBO_UTILS"])+"/overlap_wrapper_job.sh",str(self.job_id)]

        commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
        output = commandout.stdout.decode('utf-8')
        
        lines = output.split('\n')
        last_line = lines[-2] if lines else None
        if last_line:
            line_split = last_line.split()
            if len(line_split) == 1:
                return int(line_split[0])
            else:
                return -1
        else:
            return -1
        return -1    
    def makeSlurmScript(self, p_eta_point):
        p = p_eta_point[0]
        eta_min = p_eta_point[1][0]
        eta_max = p_eta_point[1][1]        
        radiator = p_eta_point[2]            
        filename = str(os.environ["AIDE_HOME"])+"/slurm_scripts/"+"jobconfig_{}_p_{}_eta_{}_{}.slurm".format(self.job_id,p,eta_min,eta_max)
        with open(filename,"w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=drich-mobo\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=1:00:00\n")
            file.write("#SBATCH --output={}/drich-mobo-subjob_%j.out\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("#SBATCH --error={}/drich-mobo-subjob_%j.err\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            
            file.write(str(os.environ["EPIC_MOBO_UTILS"])+"/shell_wrapper_job.sh {} {} {} {} {} {} \n".format(p,eta_min,eta_max,self.n_part,radiator,self.job_id))
        return filename
    def runJobs(self):
        for p_eta_point in self.p_eta_points:
            slurm_file = self.makeSlurmScript(p_eta_point)
            shellcommand = ["sbatch",slurm_file]
            commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
            output = commandout.stdout.decode('utf-8')
            line_split = output.split()
            if len(line_split) == 4:
                slurm_job_id = int(line_split[3])
                self.slurm_job_ids.append(slurm_job_id)
            else:
                #slurm job submission failed, re-submit? or just count as failed?
                self.slurm_job_ids.append(-1)
        return
    
    def get_job_status(self, slurm_id):
        # check status of slurm job        
        if slurm_id == -1:
            # something failed in job submission
            return -1
        ### HERE: run bash command to retrieve status, exit code                                                         
        shellcommand = [str(os.environ["AIDE_HOME"])+"/ProjectUtils/"+"checkSlurmStatus.sh", str(slurm_id)]
        commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
        
        output = commandout.stdout.decode('utf-8')        
        line_split = output.split()

        if len(line_split) == 1:
            status = line_split[0]
        else:
            #something wrong, try again                                                                                  
            print("Error in checking slurm status, assuming still running")
            return 0

        #  0 - running
        #  1 - completed
        # -1 - failed
        if status == "0":
            return 0
        elif status == "1":
            return 1
        elif status == "-1":
            return -1
        return 0

    def monitorJobs(self):
        # loop to check every minute if all sub-jobs are finished 
        complete = False
        while not complete:
            statuses = []
            allDone = True
            for slurm_id in self.slurm_job_ids:
                status = self.get_job_status(slurm_id)
                statuses.append(status)
                if status == 0:
                    allDone = False
            if allDone == True:
                print("Jobs finished, final statuses: ", statuses)
                complete = True
                self.final_job_status = statuses
            else:
                time.sleep(30)
        return
    def writeFailedObjectives(self):
        # executed when we have overlaps and want to punish this result,
        # but the trial didn't exactly "fail"
        # TODO: is this how we want to treat this?
        final_results = np.array( [0, 0] )
        np.savetxt(self.outname,final_results)
        return
    def retrieveResults(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives
        results = []
        result_p = []
        for i in range(len(self.p_eta_points)):
            p_eta_point = self.p_eta_points[i]
            p = p_eta_point[0]
            eta_min = p_eta_point[1][0]
            eta_max = p_eta_point[1][1]
            if self.final_job_status[i] == 1:
                K_plus_cher = np.loadtxt(str(os.environ["AIDE_HOME"])+"/log/results/"+"recon_scan_{}_kaon+_p_{}_eta_{}_{}.txt".format(self.job_id,p,eta_min,eta_max))
                pi_plus_cher = np.loadtxt(str(os.environ["AIDE_HOME"])+"/log/results/"+"recon_scan_{}_pi+_p_{}_eta_{}_{}.txt".format(self.job_id,p,eta_min,eta_max))
                mean_nphot = (pi_plus_cher[0] + K_plus_cher[0])/2
                mean_sigma = (pi_plus_cher[2] + K_plus_cher[2])/2
                if mean_sigma != 0:
                    nsigma = (abs(pi_plus_cher[1] - K_plus_cher[1])*math.sqrt(mean_nphot))/mean_sigma
                else:
                    nsigma = 0
                results.append(nsigma)
                result_p.append(p)
        # mean to reduce to 2 momentum points
        # TODO: write a function to do this based on some weighting from physics multiplicities
        result_p = np.array(result_p)
        results = np.array(results)
        
        if np.sum(result_p==14) > 0 and np.sum(result_p==40) > 0:
            final_results = np.array( [ np.mean( results[np.where(result_p==14)]),
                                        np.mean( results[np.where(result_p==40)]) ] )
        else:
            # if all jobs failed for a momentum point, exit failed
            # TODO: is this how we want to treat this case?
            sys.exit(1)
        np.savetxt(self.outname,final_results)
        return

nobj = 6
npart = 100
p_eta_scan = [
    [14, [1.3,2.0], 0],
    [14, [2.0,2.5], 0],
    [14, [2.5,3.5], 0],
    [40, [1.3,2.0], 1],
    [40, [2.0,2.5], 1],
    [40, [2.5,3.5], 1]
]

jobid = sys.argv[1]

manager = SubJobManager(p_eta_scan, npart, jobid)
noverlaps = manager.checkOverlap()

if noverlaps != 0:
    # OVERLAP OR ERROR, return -1 for all objectives
    print(noverlaps, " overlaps found, exiting trial")
    results = np.array( [-1 for i in range(len(p_eta_scan))] )
    np.savetxt(manager.outname,results)
    manager.writeFailedObjectives()
    sys.exit(0)

print("no overlaps, starting momentum/eta scan jobs")

manager.runJobs()
manager.monitorJobs()
if np.sum(manager.final_job_status) <= 0:
    manager.writeFailedObjectives()
    print("something wrong with all jobs")
else:
    manager.retrieveResults()
    print("successfully retrieved results")
