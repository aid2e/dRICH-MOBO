import os, sys, subprocess, time, uproot, math, numpy as np, awkward as ak, xml.etree.ElementTree as ET
#NEEDS TO:
# 1. run overlap check
# 2. generate p scan
# 3. calculate mu-pi sep
# 4. store all relevant results in klm-mobo-out_{jobid}.txt

class SubJobManager:
    def __init__(self, p_points, n_part, job_id,run_neutron_objectives):
        self.p_points = p_points
        self.n_part = n_part
        self.job_id = job_id
        self.outname = str(os.environ["AIDE_HOME"])+"/log/results/"+ "klm-mobo-out_{}.txt".format(jobid)
        self.slurm_job_ids = []
        self.rocauc_slurm_job_ids = []
        self.theta_min = 80
        self.theta_max = 100
        self.run_neutron_objectives = run_neutron_objectives
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
    
    def makeSlurmScript(self):       
        filename = str(os.environ["AIDE_HOME"])+"/slurm_scripts/"+"jobconfig_{}.slurm".format(self.job_id)
        with open(filename,"w") as file:
            file.write("#!/bin/bash\n")
            file.write(f"#SBATCH --job-name=submit-workflow-klm-mobo-{self.job_id}\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=8:00:00\n")
            file.write("#SBATCH --output={}/klm-mobo-subjob_%x.out\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("#SBATCH --error={}/klm-mobo-subjob_%x.err\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            DETECTOR_PATH = os.environ['DETECTOR_PATH']
            DETECTOR_CONFIG = os.environ['DETECTOR_CONFIG']
            compactFileName = f"{DETECTOR_PATH}/{DETECTOR_CONFIG}_{self.job_id}.xml"
            MOBO_path = os.environ['AIDE_HOME']
            loadEpicPath = os.environ['AIDE_HOME'] + "/load_epic.sh"
            setupPath = os.environ['AIDE_HOME'] + "/setup.sh"
            workEicPath = os.environ['WORK_EIC']
            file.write(f"source {workEicPath}/setup.sh\n")
            file.write(f"python3 {workEicPath}/slurm/submit_workflow.py --compactFile {compactFileName} --setupPath {setupPath} --loadEpicPath {loadEpicPath} --run_name_pref April_2_mobo_{self.job_id} --outFile {self.outname} --runNum {self.job_id} --chPath {MOBO_path} --deleteDfs True --no-saveGif")
        return filename
    def makeSlurmScript_mupi(self, p_point):
        p = p_point           
        filename = str(os.environ["AIDE_HOME"])+"/slurm_scripts/"+"jobconfig_{}_p_{}.slurm".format(self.job_id,p)
        with open(filename,"w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=mu_pi_{}-klm-mobo\n".format(self.job_id))
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=8:00:00\n")
            file.write("#SBATCH --output={}/%x.out\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("#SBATCH --error={}/%x.err\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            
            file.write(str(os.environ["EPIC_MOBO_UTILS"])+"shell_wrapper_job.sh {} {} {} {} {} \n".format(p,self.n_part,self.theta_min,self.theta_max,self.job_id))
        return filename
    
    def runJobs(self):
        #mu pi separation jobs
        for p_point in self.p_points:
            slurm_file = self.makeSlurmScript_mupi(p_point)                
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
        if(self.run_neutron_objectives):
            print("running neutron objective job")
            # neutral hadron energy resolution jobs
            slurm_file = self.makeSlurmScript()                
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
    
    def create_ROCAUC_job_file(self):
        filename = str(os.environ["AIDE_HOME"])+"/slurm_scripts/"+"ROCAUC_{}.slurm".format(self.job_id)
        p_points_string = ""
        for i, p in enumerate(self.p_points):
            if(self.final_job_status[i] == 1):
                p_points_string += f"{p} "
        if(p_points_string == ""):
            manager.writeFailedObjectives()
        with open(filename, "w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=ROCAUC_{}_klm-mobo\n".format(self.job_id))
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=8:00:00\n")
            file.write("#SBATCH --output={}/%x.out\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("#SBATCH --error={}/%x.err\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("cd {}\n".format(os.environ['AIDE_HOME']))
            file.write("cat << EOF | {}/eic-shell\n".format(os.environ['EIC_SHELL_HOME']))
            file.write("source ./setup.sh\n")
            file.write("source ./load_epic.sh\n")
            file.write("python3 " + str(os.environ["EPIC_MOBO_UTILS"])+"ROCAUC.py {} {} {} {} \n".format(self.n_part,self.job_id,self.outname,p_points_string))
            file.write("EOF\n")
        return filename
    def runJobs_ROCAUC(self):
        ROCAUC_file_name = self.create_ROCAUC_job_file()                
        shellcommand = ["sbatch",ROCAUC_file_name]                
        commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
        output = commandout.stdout.decode('utf-8')
        line_split = output.split()
        if len(line_split) == 4:
            slurm_job_id = int(line_split[3])
            self.rocauc_slurm_job_ids.append(slurm_job_id)
        else:
            #slurm job submission failed, re-submit? or just count as failed?
            self.rocauc_slurm_job_ids.append(-1)
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
    def monitorROCAUCJob(self):
        complete = False
        while not complete:
            statuses = []
            allDone = True
            for slurm_id in self.rocauc_slurm_job_ids:
                status = self.get_job_status(slurm_id)
                statuses.append(status)
                if status == 0:
                    allDone = False
            if allDone == True:
                print("Finished evaluating ROC Curve AUC, final statuses: ", statuses)
                complete = True
                self.rocauc_final_job_status = statuses
            else:
                time.sleep(30)
    def writeFailedObjectives(self):
        raise Exception("Writing failed objectives error")
    
    def retrieveResults(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives

        if self.final_job_status[i] == 1:
            with open(self.outname) as f:
                low_RMSE = float(f.readline().strip())
                high_RMSE = float(f.readline().strip())
            if((low_RMSE < 0) or (high_RMSE < 0)):
                manager.writeFailedObjectives()
            else:
                print(f"Results successfully aquired\n low RMSE: {low_RMSE}; high RMSE: {high_RMSE}")
        else:
            sys.exit(1)
        return
    
    # NOTE: This function is now used in ROCAUC.py, not here. Kept here for clarity
    def retrieveResults_mupi(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives
        roc_scores = []
        
        for i, p_point in enumerate(self.p_points):
            if self.final_job_status[i] == 1:
                roc_score = self.calcROCauc(
                    mu_f = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{self.job_id}_mu-_p_{p_point}.edm4hep.root'),
                    pi_f = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{self.job_id}_pi-_p_{p_point}.edm4hep.root')
                )
                roc_scores.append(roc_score)
        
        if len(roc_scores) == len(self.p_points):
            final_results = np.array(roc_scores)
        else:
            # if all jobs failed for a momentum point, exit failed
            # TODO: is this how we want to treat this case?
            sys.exit(1)
        with open(self.outname, "a") as f:
#             f.write(f"\n{final_results[0]}")
            for result in final_results:
                f.write(f"\n{result}")
#             print(f"Writing roc scores: {final_results}")
        return

    
    


if __name__ == '__main__':

    npart = 500
    p_scan = [1,5]
    
    #FOR DEBUGGING (should be true)
    run_neutron_objectives = True
    delete_root_files = True
    run_root_files = True
    #DEBUGGING SETTINGS END
    
    # format momenta into strings
    for i, p in enumerate(p_scan):
        p_scan[i] = str(int(p)) if type(p) == int or p.is_integer() else str(p)

    jobid = sys.argv[1]

    manager = SubJobManager(p_scan, npart, jobid,run_neutron_objectives)
    noverlaps = manager.checkOverlap()

    if noverlaps != 0:
        # OVERLAP OR ERROR, return -1 for all objectives
        print(noverlaps, " overlaps found, exiting trial")
        # results = np.array( [-1 for i in range(len(p_scan))] )
        # np.savetxt(manager.outname,results)
        manager.writeFailedObjectives()
        sys.exit(0)

    print("no overlaps, starting momentum scan jobs")

    if(run_root_files):
        manager.runJobs()
        manager.monitorJobs()
    else:
        manager.final_job_status = [1,1]
    if np.sum(manager.final_job_status) <= 0:
        manager.writeFailedObjectives()
        print("something wrong with all jobs")
    else:
        if(manager.run_neutron_objectives):
            manager.retrieveResults()
        manager.runJobs_ROCAUC()
        manager.monitorROCAUCJob()
        print("successfully retrieved results")
    if(delete_root_files == True):
        for p in p_scan:
            for particle in ['mu-', 'pi-']:
                filename = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{jobid}_{particle}_p_{p}.edm4hep.root')
                if os.path.exists(filename):
                    os.remove(filename)
        
