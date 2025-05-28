import numpy as np
import os, sys, subprocess, time
import math
import uncertainties

# NEEDS TO:
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
    def makeSlurmScript(self, p_eta_point, particle):
        p = p_eta_point[0]
        eta_min = p_eta_point[1][0]
        eta_max = p_eta_point[1][1]        
        radiator = p_eta_point[2]            
        filename = str(os.environ["AIDE_HOME"])+"/slurm_scripts/"+"jobconfig_{}_{}_p_{}_eta_{}_{}.slurm".format(self.job_id,particle,p,eta_min,eta_max)
        with open(filename,"w") as file:
            # currently configured for Duke computing cluster.
            # TODO: should be set somewhere in a config file
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=drich-mobo\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=scavenger\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --cpus-per-task=1\n")
            file.write("#SBATCH --mem=3G\n")
            file.write("#SBATCH --time=3:45:00\n")
            file.write("#SBATCH --output={}/drich-mobo-subjob_%j.out\n".format("log/job_output/"))
            file.write("#SBATCH --error={}/drich-mobo-subjob_%j.err\n".format("log/job_output/"))            
            file.write(str(os.environ["EPIC_MOBO_UTILS"])+"/shell_wrapper_job.sh {} {} {} {} {} {} {} \n".format(p,eta_min,eta_max,self.n_part,radiator,self.job_id,particle))
        return filename
    def runJobs(self):
        for p_eta_point in self.p_eta_points:
            for particle in ["pi+","kaon+"]:
                slurm_file = self.makeSlurmScript(p_eta_point,particle)
                shellcommand = ["sbatch",slurm_file]                
                commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
                output = commandout.stdout.decode('utf-8')
                line_split = output.split()
                if len(line_split) == 4:
                    slurm_job_id = int(line_split[3])
                    self.slurm_job_ids.append(slurm_job_id)
                else:
                    #if slurm job submission failed
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
        final_results = np.array( [0, 0, 0, 0, 0, 0] )
        np.savetxt(self.outname,final_results)
        return
    def retrieveResults(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives
        results_nsigma = []
        results_eff = []
        result_p = np.zeros(len(self.p_eta_points))
        result_etalow = np.zeros(len(self.p_eta_points))
        for i in range(len(self.p_eta_points)):
            p_eta_point = self.p_eta_points[i]
            print("p_eta_point: ", p_eta_point)
            p = p_eta_point[0]
            eta_min = p_eta_point[1][0]
            eta_max = p_eta_point[1][1]
            if self.final_job_status[i] == 1:
                print("job status 1")
                K_plus_cher = np.loadtxt(str(os.environ["AIDE_HOME"])+"/log/results/"+"recon_scan_{}_{}_kaon+_p_{}_eta_{}_{}.txt".format(self.n_part,self.job_id,p,eta_min,eta_max))
                pi_plus_cher = np.loadtxt(str(os.environ["AIDE_HOME"])+"/log/results/"+"recon_scan_{}_{}_pi+_p_{}_eta_{}_{}.txt".format(self.n_part,self.job_id,p,eta_min,eta_max))
                mean_nphot = (pi_plus_cher[0] + K_plus_cher[0])/2
                mean_sigma = (pi_plus_cher[2] + K_plus_cher[2])/2
                #calculate nsigma separation 
                if mean_sigma != 0:
                    nsigma = (abs(pi_plus_cher[1] - K_plus_cher[1])*math.sqrt(mean_nphot))/mean_sigma
                else:
                    nsigma = 0
                #get mean fraction of tracks with reco photons
                mean_eff = (pi_plus_cher[3] + K_plus_cher[3])/2
                results_nsigma.append(uncertainties.ufloat(nsigma,p_eta_point[3]*nsigma))
                results_eff.append(uncertainties.ufloat(mean_eff,p_eta_point[4]*mean_eff))
                result_p[i] = p
                result_etalow[i] = eta_min
        results_nsigma = np.array(results_nsigma)
        results_eff = np.array(results_eff)
        
        piKsep_etalow = np.mean(results_nsigma[result_p==15])
        piKsep_etahigh = np.mean(results_nsigma[result_p==45])
        acc_total = np.mean(results_eff[1:])
        
        final_results = np.array( [ piKsep_etalow.n, 0,
                                    piKsep_etahigh.n, 0,
                                    acc_total.n, 0
                                   ]
                                )
        if np.any(np.isnan(final_results)):
            # flag as failed if we got any NaN objective somehow,
            # otherwise fitting the surrogate model will fail
            sys.exit(1)
        np.savetxt(self.outname,final_results)
        return
    def checkAcceptance(self):
        # Script to check small set of simulated particles to ensure acceptance is reasonable,
        # consider early-stopping a bad geometry
        results_eff = []
        result_p = []
        result_etalow = []
        for i in range(len(self.p_eta_points)):
            p_eta_point = self.p_eta_points[i]
            print("p_eta_point: ", p_eta_point)
            p = p_eta_point[0]
            eta_min = p_eta_point[1][0]
            eta_max = p_eta_point[1][1]
            if self.final_job_status[i] == 1:
                print("job status 1")
                K_plus_cher = np.loadtxt(str(os.environ["AIDE_HOME"])+"/log/results/"+"recon_scan_{}_{}_kaon+_p_{}_eta_{}_{}.txt".format(self.n_part,self.job_id,p,eta_min,eta_max))
                pi_plus_cher = np.loadtxt(str(os.environ["AIDE_HOME"])+"/log/results/"+"recon_scan_{}_{}_pi+_p_{}_eta_{}_{}.txt".format(self.n_part,self.job_id,p,eta_min,eta_max))
        
                mean_eff = (pi_plus_cher[3] + K_plus_cher[3])/2
                results_eff.append(mean_eff)
                result_p.append(p)
                result_etalow.append(eta_min)
        result_p = np.array(result_p)
        result_etalow = np.array(result_etalow)
        results_eff = np.array(results_eff)        

        acc_all = np.mean( results_eff )        
        return acc_all

npart = 1000
# [momentum, eta range, radiator, percent_dev_piKsep, percent_dev_acc]
#TODO: update uncertainty estimates
p_eta_scan = [    
    #[6, [1.4,1.5], 0, 0.0, 0.0],
    [15, [1.5,2.0], 0, 0.0, 0.0],
    [15, [2.0,2.5], 0, 0.0, 0.0],
    [45, [2.5,3.0], 1, 0.0, 0.0],
    [45, [3.0,3.5], 1, 0.0, 0.0]
]

p_eta_scan_init = [    
    [40, [1.5,2.0], 1],
    [40, [2.0,2.5], 1],
    [40, [2.5,3.5], 1]
]

jobid = sys.argv[1]

manager = SubJobManager(p_eta_scan, npart, jobid)
# ADD BACK IN
noverlaps = manager.checkOverlap()

if noverlaps != 0:
    # OVERLAP OR ERROR, flag trial as failed to Ax
    print(noverlaps, " overlaps found, exiting trial")
    # flag trial as failure
    sys.exit(1)

print("no overlaps, starting momentum/eta scan jobs")
# run an initial scan of ~100 particles to check that acceptance is okay
'''
manager_test = SubJobManager(p_eta_scan_init, 100, jobid)
manager_test.runJobs()
manager_test.monitorJobs()
avgAcc = manager_test.checkAcceptance()

# if average acceptance less than 35%, early stop this design
if avgAcc < 0.35:
    print("bad preliminary acceptance test, flag as failure and write out failed objectives")
    manager_test.writeFailedObjectives()
    sys.exit(0)
'''
# if results were okay, finish analyzing this design point
manager.runJobs()
manager.monitorJobs()

manager.retrieveResults()
