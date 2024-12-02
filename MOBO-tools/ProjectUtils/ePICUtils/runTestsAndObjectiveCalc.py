import os, sys, subprocess, time, uproot, math, numpy as np, awkward as ak, ROOT

#NEEDS TO:
# 1. run recon-util
# 2. calculate mean of mchi2
# 3. store all relevant results in rich-global-mobo-out_{jobid}.txt

class SubJobManager:
    def __init__(self, p_points, job_id):
        self.p_points = p_points
        self.job_id = job_id
        self.outname = str(os.environ["AIDE_HOME"])+"/rich/log/results/"+ "rich-global-mobo-out_{}.txt".format(jobid)
        self.slurm_job_ids = []
            
    def makeSlurmScript(self, p_point):
        p = p_point           
        filename = str(os.environ["AIDE_HOME"])+"/rich_slurm_scripts/"+"jobconfig_{}_module_{}.slurm".format(self.job_id,p)
        with open(filename,"w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=rich-global-mobo\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=2:00:00\n")
            file.write("#SBATCH --output={}/rich-global-mobo-subjob_%j.out\n".format(str(os.environ["AIDE_HOME"])+"/rich/log/job_output"))
            file.write("#SBATCH --error={}/rich-global-mobo-subjob_%j.err\n".format(str(os.environ["AIDE_HOME"])+"/rich/log/job_output"))
            file.write("setenv CCDB_CONNECTION:///"+str(os.environ["AIDE_HOME"])+"/ccdb_2024-11-03.sqlite\n") 
            file.write("ccdb mkvar variation_{}\n".format(self.job_id))
            file.write("ccdb add /geometry/rich/module1/alignment -v variation_{} variation_{}.dat\n".format(self.job_id, self.job_id))
            file.write("recon-util -y rich_{}.yaml -i input.hipo -o "+str(os.environ["AIDE_HOME"])+"/rich/log/hipo_files/output_{}_module_{}.hipo\n".format(self.job_id, self.job_id, p))
            file.write(str(os.environ["AIDE_HOME"])+"/RICH-track-matching-tree "+str(os.environ["AIDE_HOME"])+"/rich/log/root_files/output_{}_module_{}.root "+str(os.environ["AIDE_HOME"])+"/rich/log/hipo_files/output_{}_module_{}.hipo\n".format(self.job_id, p, self.job_id, p)) 
            
        return filename
    
    def runJobs(self):
        for p_point in self.p_points:
            slurm_file = self.makeSlurmScript(p_point)                
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
        raise Exception("Writing failed objectives error")
    
    def retrieveResults(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives

        for p_point in self.p_points:
            if self.final_job_status[i] == 1:
                mean_mchi2_value = self.calcMeanmchi2(
                   mchi2_f = os.path.join(os.environ['AIDE_HOME'], f'rich/log/root_files/output_{self.job_id}_module_{p_point}.root'))
                )
                with open(self.outname, 'w') as file:
                    file.write(f"{mean_mchi2_value}")
                
        return
    
    def calcMeanmchi2(self, mchi2_f):
        root_file = ROOT.TFile(mchi2_f)
        tree = mchi2_f.Get("RICHTree")
        branch = tree.GetBranch("mchi2")
        hist = ROOT.TH1F("hist", "Histogram of mchi2", 60, 0, 30) 

        # Loop over the entries in the tree
        for entry in tree:
            value = getattr(entry, "mchi2") 
            hist.Fill(value)

        # Calculate the mean value of mchi2
        mean_value = hist.GetMean()

        return mean_value

if __name__ == '__main__':

    p_scan = [1]

    # format momenta into strings
    for i, p in enumerate(p_scan):
        p_scan[i] = str(int(p)) if type(p) == int or p.is_integer() else str(p)

    jobid = sys.argv[1]

    manager = SubJobManager(p_scan, jobid)
    
    manager.runJobs()
    manager.monitorJobs()
    if np.sum(manager.final_job_status) <= 0:
        manager.writeFailedObjectives()
        print("something wrong with all jobs")
    else:
        manager.retrieveResults()
        print("successfully retrieved results")
        
    #for p in p_scan:
        #filename = os.path.join(os.environ['AIDE_HOME'], f'rich/log/hipo_files/scan_{jobid}_module_{p}.hipo')
        #if os.path.exists(filename):
            #os.remove(filename)
