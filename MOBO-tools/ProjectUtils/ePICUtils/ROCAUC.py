'''
How to test mu/pi separation (without running mobo)
1. source ~/.mobo-bashrc
2. conda activate dRICH-MOBO
2.5 cd MOBO-tools
3. source MOBO-tools/setup.sh
3.5 set flags in main function of newRunTestsAndObjectives.py
4. python3 ProjectUtils/ePICUtils/newRunTestsAndObjectives.py 10001
   1. use the jobid of whichever geometry you want to use (klmws_only_0.xml)

'''

import os, sys, subprocess, time, uproot, math, numpy as np, awkward as ak, xml.etree.ElementTree as ET
from sklearn.metrics import roc_auc_score,roc_curve
import dd4hep
import ROOT
from collections import defaultdict
import matplotlib.pyplot as plot
class ROCAUC_calc:
    def __init__(self, p_points, n_part, job_id,plot_roc_curves = False, plot_path = "./"):
        self.p_points = p_points
        self.n_part = n_part
        self.job_id = job_id
        self.plot_roc_curves = plot_roc_curves
        self.plot_path = plot_path
        self.outname = str(os.environ["AIDE_HOME"])+"/log/results/"+ "klm-mobo-out_{}.txt".format(jobid)
        self.slurm_job_ids = []
        if(self.job_id < 10000):
            self.load_geometry("{}/{}_{}.xml".format(os.environ["DETECTOR_PATH"],os.environ["DETECTOR_CONFIG"],self.job_id))
        else:
            self.load_geometry("{}/{}.xml".format(os.environ["DETECTOR_PATH"],os.environ["DETECTOR_CONFIG"]))
        self.theta_min = 80
        self.theta_max = 100
        self.barrel_length = 1500 # length of the barrel along the z-axis
        self.barrel_offset = 18 # offset of the barrel in the positive z direction
    def load_geometry(self,compactFile):
        lcdd = dd4hep.Detector.getInstance()
        lcdd.fromXML(compactFile)
        print(f"loaded compact file: {compactFile}")
        self.lcdd = lcdd
        if(self.job_id < 10000):
            root = ET.parse("{}/compact/pid/klmws_{}.xml".format(os.environ['DETECTOR_PATH'],self.job_id)).getroot()
        else:
            root = ET.parse("{}/compact/pid/klmws.xml".format(os.environ['DETECTOR_PATH'],self)).getroot()
        self.n_layers = int(root.find(".//constant[@name='HcalScintillatorNbLayers']").get('value')) # number of superlayers
    
    
    # hit is object from looping over root_file
    def get_layer_number(self, hit):
        cellID = hit.cellID
    #     print(f"CellID: {cellID}")

        # Get the IDDescriptor for the HcalBarrel
        id_spec = self.lcdd.idSpecification("HcalBarrelHits")
        if not id_spec:
            print("Failed to get IDSpecification for HcalBarrelHits")
            return None

        id_dec = id_spec.decoder()

        # Extract individual field values
        try:
            layer = id_dec.get(cellID, "layer")
        except Exception as e:
            print(f"Error decoding cellID: {e}")
            return None

        return layer - 1
    # returns the number of pixels detected by a hit
    def pixel_num(self, energy_dep, zpos):
        inverse = lambda x : 4.9498 / (29.9733 - x + self.barrel_length / 2) - 0.0016796
        efficiency = inverse(zpos - self.barrel_offset) + inverse(self.barrel_offset - zpos) # ratio of photons produced in a hit that make it to the sensor
        return 10 * energy_dep * (1000 * 1000) * efficiency * 0.5

    def layer_calc_cellID(self,root_file_path,particle):
        root_file = ROOT.TFile(root_file_path)
        tree = root_file.Get("events")
        max_layer_numbers = []
        max_rpositions = []
        num_pixels_per_layer = np.zeros((self.n_part,self.n_layers))
        for event_idx, event in enumerate(tree):
            event_layer_numbers = []
            rpositions = []
            for hit_idx, hit in enumerate(event.HcalBarrelHits):
                layer_num = self.get_layer_number(hit)
                event_layer_numbers.append(layer_num)
                zpos = hit.position[2]
                rpos = np.sqrt(hit.position[1] ** 2 + hit.position[0] ** 2)
                rpositions.append(rpos)
                EDep = hit.EDep
                num_pixels = self.pixel_num(EDep,zpos)
                if(layer_num >=0 and layer_num <(self.n_layers)):
                    num_pixels_per_layer[event_idx,layer_num] += num_pixels
                else:
                    print(f"Found hit #{hit_idx} in event #{event_idx} to be in layer #{layer_num}, which does not fit compact file definition. Skipping hit...")
            if(len(event_layer_numbers) > 0):
                max_layer_numbers.append(np.max(np.array(event_layer_numbers)))
            if(len(rpositions) > 0):
                max_rpositions.append(np.max(np.array(rpositions)))
        fig,axs = plot.subplots(1,2,figsize = (10,5))
        axs[0].scatter(range(len(max_rpositions)),max_rpositions)
        axs[0].set_xlabel(f"{particle}")
        axs[1].hist(max_rpositions)
        axs[1].set_xlabel(f"{particle}")
        fig.savefig(f"test_{particle}.jpeg")

        #get mask for valid hit layers
        mask = num_pixels_per_layer >= 2

        layers_traveled = np.empty(num_pixels_per_layer.shape[0], dtype=int)
        layer_counts = np.zeros(self.n_layers)
        
        # For each event, get furthest layer hit
        for i, row_mask in enumerate(mask):
            indices = np.where(row_mask)[0]
            #get last index
            layers_traveled[i] = indices[-1] if indices.size > 0 else -1
            layer_counts[layers_traveled[i]] += 1
        return layers_traveled, layer_counts

    # return the area under the ROC curve for discriminating between muons and pions
    def calcROCauc(self, mu_f, pi_f,p):
        mu_layers_traveled, mu_layer_counts = self.layer_calc_cellID(mu_f,"mu")
        pi_layers_traveled, pi_layer_counts = self.layer_calc_cellID(pi_f,"pi")

        layers_traveled_tot = np.concatenate((mu_layers_traveled, pi_layers_traveled))
        # 1 if particle is muon, 0 if particle is pion
        pid_actual = np.concatenate((np.ones_like(mu_layers_traveled), np.zeros_like(pi_layers_traveled)))
        # probability that a particle stopping at this layer is a muon
        pid_layer_prob = np.divide(mu_layer_counts, mu_layer_counts + pi_layer_counts, out=np.zeros(mu_layer_counts.size), where=(mu_layer_counts+pi_layer_counts)!=0)
        # probability that each particle is a muon
        pid_model = pid_layer_prob[layers_traveled_tot]

        # calculate ROC AUC score
        roc_score = roc_auc_score(pid_actual, pid_model)
        if(self.plot_roc_curves):
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(pid_actual, pid_model)
            print(f"fpr: {fpr}")
            print(f"tpr: {tpr}")
            print(f"thresholds: {thresholds}")

            # Create the plot
            plot.figure()
            plot.plot(fpr, tpr, color='blue', lw=2, 
                    label=f'ROC curve (area = {roc_score:.2f})')
#             plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
#                     label='Random classifier')
            plot.xlim([0.0, 1.0])
            plot.ylim([0.0, 1.05])
            plot.xlabel('False Positive Rate', fontsize = 20)
            plot.ylabel('True Positive Rate', fontsize = 20)
#             plot.title('Receiver Operating Characteristic (ROC) Curve')
            plot.legend(loc="lower right",fontsize = 20)
            plot.grid(True)
            plot.savefig(self.plot_path + f"roc_curve_{p}GeV.pdf")
        return roc_score
    def retrieveResults_mupi(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives
        roc_scores = []
        
        for i, p_point in enumerate(self.p_points):
            roc_score = self.calcROCauc(
                mu_f = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{self.job_id}_mu-_p_{p_point}.edm4hep.root'),
                pi_f = os.path.join(os.environ['AIDE_HOME'], f'log/sim_files/scan_{self.job_id}_pi-_p_{p_point}.edm4hep.root'),
                p = p_point
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
            print(f"Writing roc scores: {final_results}")
        return
    
if __name__ == '__main__':
    n_part = int(sys.argv[1])
    jobid = int(sys.argv[2])
    outname = sys.argv[3]
    p_points = []
    for i, arg in enumerate(sys.argv):
        if(i <=5):
            continue
        else:
            p_points.append(int(sys.argv[i]))
    rocauc_calc = ROCAUC_calc(p_points, n_part, jobid,plot_roc_curves = sys.argv[4], plot_path = sys.argv[5])
#     try:
    rocauc_calc.retrieveResults_mupi()
    print("Successfully calculated ROC AUC and wrote to result file")
#     except Exception as e:
#         print(f"AUC calc failed with exception {e}")