import os, sys, subprocess, time, uproot, math, numpy as np, awkward as ak, xml.etree.ElementTree as ET
from sklearn.metrics import roc_auc_score

#NEEDS TO:
# 1. run overlap check
# 2. generate p scan
# 3. calculate mu-pi sep
# 4. store all relevant results in klm-mobo-out_{jobid}.txt

class SubJobManager:
    def __init__(self, p_points, n_part, job_id):
        self.p_points = p_points
        self.n_part = n_part
        self.job_id = job_id
        self.outname = str(os.environ["AIDE_HOME"])+"/log/results/"+ "klm-mobo-out_{}.txt".format(jobid)
        self.slurm_job_ids = []
        self.calcGeomVals()
        
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
            file.write(f"#SBATCH --job-name=klm-mobo-{self.job_id}\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=scavenger-gpu\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=1:00:00\n")
            file.write("#SBATCH --output={}/klm-mobo-subjob_%j.out\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("#SBATCH --error={}/klm-mobo-subjob_%j.err\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            
            file.write(f"python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/submit_workflow.py --run_name_pref March10_mobo_test_{self.job_id} --outFile {self.outname}")
        return filename
    def makeSlurmScript_mupi(self, p_point):
        p = p_point           
        filename = str(os.environ["AIDE_HOME"])+"/slurm_scripts/"+"jobconfig_{}_p_{}.slurm".format(self.job_id,p)
        with open(filename,"w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --job-name=klm-mobo\n")
            file.write("#SBATCH --account=vossenlab\n")
            file.write("#SBATCH --partition=common\n")
            file.write("#SBATCH --mem=2G\n")
            file.write("#SBATCH --time=2:00:00\n")
            file.write("#SBATCH --output={}/klm-mobo-subjob_mu_pi_%j.out\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            file.write("#SBATCH --error={}/klm-mobo-subjob_mu_pi_%j.err\n".format(str(os.environ["AIDE_HOME"])+"/log/job_output"))
            
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
#         As = []

        if self.final_job_status[i] == 1:
            with open(self.outname) as f:
#                     A = float(f.readline().strip())
                RMSE = float(f.readline().strip())
            print(f"Results successfully aquired\n RMSE: {RMSE}")
#                 print(f"Results successfully aquired\n RMSE: {RMSE}; A: {A}")
        else:
            sys.exit(1)
        return

    def retrieveResults_mupi(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives
        roc_scores = []
        
        for p_point in self.p_points:
            if self.final_job_status[i] == 1:
                roc_score = self.calcROCauc(
                    mu_f = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{self.job_id}_mu-_p_{p_point}.edm4hep.root'),
                    pi_f = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{self.job_id}_pi-_p_{p_point}.edm4hep.root')
                )
                roc_scores.append(roc_score)
        
        if len(roc_scores) == len(self.p_points):
            final_results = np.array([roc_scores[0], roc_scores[1]])
        else:
            # if all jobs failed for a momentum point, exit failed
            # TODO: is this how we want to treat this case?
            sys.exit(1)
#         np.savetxt(self.outname,final_results)
        with open(self.outname, "a") as f:
            f.write(f"\n{final_results[0]}\n{final_results[1]}\n{self.outer_radius}")
            print(f"Writing roc scores: {final_results} and outer radius: {self.outer_radius}")
        return
    
    def calcGeomVals(self):
        # lengths in mm
        # superlayer geom:
        # steel layer | air gap 1 | scintillating layer 1 | air gap 2 | air gap 3 | scintillating layer 2 | air gap 4
        
        xmlfile = os.path.join(os.environ['EPIC_HOME'], f'compact/pid/klmws_{self.job_id}.xml')
        root = ET.parse(xmlfile).getroot()

        self.superlayer_count = int(root.find(".//constant[@name='HcalScintillatorNbLayers']").get('value')) # number of superlayers
        self.steel_thick = float(root.find(".//constant[@name='HcalSteelThickness']").get('value')[:-3]) # thickness of steel sublayer
        self.sens_sublayer_thick = float(root.find(".//constant[@name='HcalScintillatorThickness']").get('value')[:-3]) # thickness of sensitive sublayer
        
        self.inner_radius = 1770 # starting radial position of first layer
        self.air_gap_thick = 0.3 # thickness of air gap between sublayers
        self.superlayer_dist = self.steel_thick + self.sens_sublayer_thick + self.air_gap_thick * 2 # width of superlayer
        
        self.outer_radius = self.inner_radius + self.superlayer_count * self.superlayer_dist # outer radius of barrel
        
        self.sector_count = 8 # number of radial sectors
        self.barrel_length = 1500 # length of the barrel along the z-axis
        self.barrel_offset = 18 # offset of the barrel in the positive z direction
        
        # min/max theta angles to shoot particle gun
        self.theta_min = 90 + np.rad2deg(np.arctan( (-self.barrel_length / 2 + self.barrel_offset) / self.outer_radius )) * 0.85
        self.theta_max = 90 + np.rad2deg(np.arctan( (self.barrel_length / 2 + self.barrel_offset) / self.outer_radius )) * 0.85

        self.first_sens_sublayer_pos = self.inner_radius + self.steel_thick + self.air_gap_thick  # position of first sensitive sublayer

        # array containing the start pos of each sensitive sublayer
        self.layer_pos = np.zeros(self.superlayer_count * 2) # 2 sensitive sublayers per superlayer
        self.layer_pos = [self.first_sens_sublayer_pos + self.superlayer_dist*i for i in range(self.superlayer_count)] # first sublayers 

    # returns the distance in the direction of the nearest sector
    def sector_proj_dist(self, xpos, ypos):
        sector_angle = (np.arctan2(ypos, xpos) + np.pi / self.sector_count) // (2*np.pi / self.sector_count) * 2*np.pi / self.sector_count # polar angle (in radians) of the closest sector
        return xpos * np.cos(sector_angle) + ypos * np.sin(sector_angle) # scalar projection of position vector onto unit direction vector 

    # returns the layer number for the position of a detector hit
    def layer_num(self, xpos, ypos):
        pos = self.sector_proj_dist(xpos, ypos)

        # false if hit position is before the first sensitive sublayer or after the last sensitive sublayer 
        within_layer_region = np.logical_and(pos * 1.0001 > self.layer_pos[0], pos / 1.0001 < self.layer_pos[-1] + self.sens_sublayer_thick)

        superlayer_index = np.where(within_layer_region, ak.values_astype( (pos * 1.0001 - self.layer_pos[0]) // self.superlayer_dist, 'int64'), -1) # index of superlayer the hit may be in, returns -1 if out of region
        layer_pos_dup = ak.Array(np.broadcast_to(self.layer_pos, (int(ak.num(superlayer_index, axis=0)), len(self.layer_pos)))) # turn layer_pos into a 2d array with duplicate rows to allow indexing
        dis_from_first_sublayer = np.where(within_layer_region, pos - layer_pos_dup[superlayer_index], -1) # distance of hit from the first sublayer in the superlayer, returns -1 if out of region

        # true if hit is within the first of the paired layers
        in_first_layer = np.logical_and(within_layer_region, dis_from_first_sublayer / 1.0001 <= self.sens_sublayer_thick)

        # layer number of detector hit; returns -1 if not in a layer
        hit_layer = np.where(in_first_layer, superlayer_index + 1, -1)
        return hit_layer

    # returns the number of pixels detected by a hit
    def pixel_num(self, energy_dep, zpos):
        inverse = lambda x : 4.9498 / (29.9733 - x + self.barrel_length / 2) - 0.0016796
        efficiency = inverse(zpos - self.barrel_offset) + inverse(self.barrel_offset - zpos) # ratio of photons produced in a hit that make it to the sensor
        return 10 * energy_dep * (1000 * 1000) * efficiency * 0.5

    # takes in position and energy deposited for a hit as ragged 2d awkward arrays
    # with each row corresponding to hits produced by a particle and its secondaries
    # returns 1d numpy array containing number of layers traveled by each track (-1: hits not in any layer, -2: hits produce too few pixels, -3: no hits for this particle)
    # and a 1d numpy array containing number of terminating tracks for each layer, starting at layer 1 
    def layer_calc(self, xpos, ypos, zpos, energy_dep):
        hit_layer = self.layer_num(xpos, ypos)
        hit_layer_filtered = np.where(self.pixel_num(energy_dep, zpos) >= 2, hit_layer, -2) # only accept layers with at least 2 pixels
        layers_traveled = ak.fill_none(ak.max(hit_layer_filtered, axis=1), -3) # max accepted layers traveled for a track determines total layers traveled
        layer_counts = np.asarray(ak.sum(layers_traveled[:, None] == np.arange(1, self.superlayer_count + 1), axis=0)) # find counts for each layer, 1 through max
        return np.asarray(layers_traveled)[layers_traveled >= 1], layer_counts

    # return the area under the ROC curve for discriminating between muons and pions
    def calcROCauc(self, mu_f, pi_f):
        with uproot.open(mu_f) as mu_file:
            mu_hit_x = mu_file['events/HcalBarrelHits.position.x'].array()
            mu_hit_y = mu_file['events/HcalBarrelHits.position.y'].array()
            mu_hit_z = mu_file['events/HcalBarrelHits.position.z'].array()
            mu_hit_edep = mu_file['events/HcalBarrelHits.EDep'].array()

        with uproot.open(pi_f) as pi_file:
            pi_hit_x = pi_file['events/HcalBarrelHits.position.x'].array()
            pi_hit_y = pi_file['events/HcalBarrelHits.position.y'].array()
            pi_hit_z = pi_file['events/HcalBarrelHits.position.z'].array()
            pi_hit_edep = pi_file['events/HcalBarrelHits.EDep'].array()

        mu_layers_traveled, mu_layer_counts = self.layer_calc(mu_hit_x, mu_hit_y, mu_hit_z, mu_hit_edep)
        pi_layers_traveled, pi_layer_counts = self.layer_calc(pi_hit_x, pi_hit_y, pi_hit_z, pi_hit_edep)

        layers_traveled_tot = np.concatenate((mu_layers_traveled, pi_layers_traveled))
        # 1 if particle is muon, 0 if particle is pion
        pid_actual = np.concatenate((np.ones_like(mu_layers_traveled), np.zeros_like(pi_layers_traveled)))
        # probability that a particle stopping at this layer is a muon
        pid_layer_prob = np.divide(mu_layer_counts, mu_layer_counts + pi_layer_counts, out=np.zeros(mu_layer_counts.size), where=(mu_layer_counts+pi_layer_counts)!=0)
        # probability that each particle is a muon
        pid_model = pid_layer_prob[layers_traveled_tot - 1]

        # calculate ROC AUC score
        roc_score = roc_auc_score(pid_actual, pid_model)

        return roc_score
    
    


if __name__ == '__main__':

    npart = 250
    p_scan = [1, 5]

    # format momenta into strings
    for i, p in enumerate(p_scan):
        p_scan[i] = str(int(p)) if type(p) == int or p.is_integer() else str(p)

    jobid = sys.argv[1]

    manager = SubJobManager(p_scan, npart, jobid)
    noverlaps = manager.checkOverlap()

    if noverlaps != 0:
        # OVERLAP OR ERROR, return -1 for all objectives
        print(noverlaps, " overlaps found, exiting trial")
        # results = np.array( [-1 for i in range(len(p_scan))] )
        # np.savetxt(manager.outname,results)
        manager.writeFailedObjectives()
        sys.exit(0)

    print("no overlaps, starting momentum scan jobs")

    manager.runJobs()
    manager.monitorJobs()
    if np.sum(manager.final_job_status) <= 0:
        manager.writeFailedObjectives()
        print("something wrong with all jobs")
    else:
        manager.retrieveResults()
        manager.retrieveResults_mupi()
        print("successfully retrieved results")
    for p in p_scan:
        for particle in ['mu-', 'pi-']:
            filename = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{jobid}_{particle}_p_{p}.edm4hep.root')
            if os.path.exists(filename):
                os.remove(filename)
        
