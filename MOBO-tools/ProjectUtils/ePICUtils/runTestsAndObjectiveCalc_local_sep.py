import base64
import json
import numpy as np
import os
import sys
import subprocess
# import math
# import uncertainties
import logging
import traceback

# NEEDS TO:
# 1. run overlap check
# 2. generate p/eta scan
# 3. calculate pi-K sep
# 4. store all relevant results in drich-mobo-out_{jobid}.txt


logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(threadName)s\t%(name)s\t%(levelname)s\t%(message)s')


class SubJobManager:
    def __init__(self, p_eta_point, particle, n_part, job_id):
        self.p_eta_point = p_eta_point
        self.p_eta_points = [p_eta_point]
        self.particle = particle
        self.particles = [particle]

        self.n_part = n_part
        self.job_id = job_id
        # self.outname = str(os.environ["AIDE_HOME"])+"/log/results/"+ "drich-mobo-out_{}.txt".format(jobid)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        logging.info(f"SubJobManager dir_path: {self.dir_path}")
        if os.environ.get("AIDE_WORKDIR", None):
            self.output_dir = os.environ.get("AIDE_WORKDIR")
        else:
            self.output_dir = os.getcwd()
            os.environ['AIDE_WORKDIR'] = self.output_dir
        self.output_name = os.path.join(self.output_dir, "log/results/drich-mobo-out_{}.npz".format(job_id))
        self.status_name = os.path.join(self.output_dir, "log/results/drich-mobo-status_{}.txt".format(job_id))
        for f in [self.output_name, self.status_name]:
            if os.path.exists(f):
                os.remove(f)

        # self.particles = ["pi+", "kaon+"]
        self.final_job_status = {}
        self.final_job_result = {}

    def checkOverlap(self):
        shellcommand = [os.path.join(self.dir_path, "overlap_wrapper_job_local.sh"), str(self.job_id)]

        logging.info("SubJobManager ++++++ checkOverlap +++++++")
        logging.info(f"SubJobManager command: {shellcommand}")
        commandout = subprocess.run(shellcommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = commandout.stdout.decode('utf-8')
        error = commandout.stderr.decode('utf-8')
        logging.info("SubJobManager output:")
        logging.info(output)
        logging.info("SubJobManager error:")
        logging.info(error)
        logging.info("SubJobManager ++++++ end checkOverlap +++++++")

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

    def runJobs(self):
        logging.info("SubJobManager ++++++ runJobs +++++++")
        num_jobs = 0
        for point_num, p_eta_point in enumerate(self.p_eta_points):
            # if p_eta_point not in self.final_job_status:
            #     self.final_job_status[p_eta_point] = {}
            self.final_job_status[point_num] = {}

            for particle in self.particles:
                # slurm_file = self.makeSlurmScript(p_eta_point,particle)
                # shellcommand = ["sbatch",slurm_file]

                p = p_eta_point[0]
                eta_min = p_eta_point[1][0]
                eta_max = p_eta_point[1][1]
                radiator = p_eta_point[2]
                shell_command = [os.path.join(self.dir_path, "shell_wrapper_job_local.sh"),
                                 str(p), str(eta_min), str(eta_max), str(self.n_part), str(radiator), self.job_id, particle]
                logging.info(f"Job No.{num_jobs} SubJobManager command: {shell_command}")
                # num_jobs += 1
                # commandout = subprocess.run(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                commandout = subprocess.run(shell_command)
                # output = commandout.stdout.decode('utf-8')
                # error = commandout.stderr.decode('utf-8')
                status_code = commandout.returncode
                logging.info(f"Job No.{num_jobs} SubJobManager returncode: {status_code}")
                self.final_job_status[point_num][particle] = int(status_code)

                output = commandout.stdout
                error = commandout.stderr
                if output:
                    output = output.decode('utf-8')
                if error:
                    error = error.decode('utf-8')
                logging.info(f"Job No.{num_jobs} SubJobManager output:")
                logging.info(output)
                logging.info(f"Job No.{num_jobs} SubJobManager error:")
                logging.info(error)
                num_jobs += 1
        logging.info("SubJobManager ++++++ end runJobs +++++++")
        return

    def writeFailedObjectives(self):
        # executed when we have overlaps and want to punish this result,
        # but the trial didn't exactly "fail"
        # TODO: is this how we want to treat this?
        final_results = np.array([0, 0, 0, 0, 0, 0])
        np.savetxt(self.output_name, final_results)
        return

    def write_status_code(self, status_code):
        with open(self.status_name, 'w') as f:
            f.write(str(status_code))

    def retrieveResults(self):
        # when results finished, retrieve analysis script outputs
        # and calculate objectives
        logging.info("SubJobManager retrieving results")

        # results = {}
        # for i in range(len(self.p_eta_points)):
        for i, p_eta_point in enumerate(self.p_eta_points):
            # p_eta_point = self.p_eta_points[i]
            logging.info(f"SubJobManager job.{i} p_eta_point: {p_eta_point}")
            p = p_eta_point[0]
            eta_min = p_eta_point[1][0]
            eta_max = p_eta_point[1][1]

            for particle in self.particles:
                plus_cher = np.loadtxt(str(os.environ["AIDE_WORKDIR"]) + "/log/results/" + "recon_scan_{}_{}_p_{}_eta_{}_{}.txt".format(self.job_id, particle, p, eta_min, eta_max))
                # self.final_job_result[str(p_eta_point)][particle] = plus_cher
                self.final_job_result[str(i)] = {'point': p_eta_point, 'particle': particle, 'plus_cher': plus_cher.tolist()}

        logging.info(f"SubJobManager saving results to {self.output_name}: {self.final_job_result}")
        np.savez(self.output_name, **self.final_job_result)
        logging.info("SubJobManager saved results")
        return

    def get_final_job_status(self):
        total, num_done = 0, 0
        logging.info(f"SubJobManager final_job_status: {self.final_job_status}")
        for i, p_eta_point in enumerate(self.p_eta_points):
            for particle in self.particles:
                total += 1
                if self.final_job_status[i][particle] == 0:
                    num_done += 1
        logging.info(f"SubJobManager final_job_status: num_done {num_done}, total: {total}")
        return num_done, total


def run(jobid, npart, p_eta_point, particle):
    # npart = 1500
    # npart = 2
    # [momentum, eta range, radiator, dev_piKsep, dev_acc]
    # std dev for 2500 tracks
    '''
    p_eta_scan = [
        [15, [1.3,2.0], 0, 0.01992527, 0.00700383],
        [15, [2.0,2.5], 0, 0.01385765, 0.00723669],
        [15, [2.5,3.5], 0, 0.01577519, 0.00987238],
        [40, [1.3,2.0], 1, 0.03229172, 0.00645222],
        [40, [2.0,2.5], 1, 0.01007051, 0.00276879],
        [40, [2.5,3.5], 1, 0.0106865, 0.00272314]
    ]
    '''

    # std dev for 2000 tracks
    '''
    p_eta_scan = [
        [15, [1.5,2.0], 0, 0.02242128, 0.00802887],
        [15, [2.0,2.5], 0, 0.01539502, 0.00788831],
        [15, [2.5,3.5], 0, 0.01757623, 0.00998055],
        [40, [1.5,2.0], 1, 0.03879622, 0.00747648],
        [40, [2.0,2.5], 1, 0.01182995, 0.00298889],
        [40, [2.5,3.5], 1, 0.01211408, 0.00309565]
    ]
    '''

    '''
    # std dev for 1500 tracks
    p_eta_scan = [
        [15, [1.5, 2.0], 0, 0.02457205, 0.00878185],
        [15, [2.0, 2.5], 0, 0.01871374, 0.00916126],
        [15, [2.5, 3.5], 0, 0.02046443, 0.01240257],
        [40, [1.5, 2.0], 1, 0.04405797, 0.00824205],
        [40, [2.0, 2.5], 1, 0.01390604, 0.00350744],
        [40, [2.5, 3.5], 1, 0.01391206, 0.00349814]
    ]
    '''

    manager = SubJobManager(p_eta_point, particle, npart, jobid)
    noverlaps = manager.checkOverlap()

    if noverlaps != 0:
        # OVERLAP OR ERROR, return -1 for all objectives
        logging.info(f"noverlaps: {noverlaps}, overlaps found, exiting trial")
        # results = np.array([-1 for i in range(len(p_eta_scan))])
        # np.savetxt(manager.output_name, results)
        manager.writeFailedObjectives()
        sys.exit(0)

    logging.info("no overlaps, starting momentum/eta scan jobs")

    manager.runJobs()
    # manager.monitorJobs()

    manager.retrieveResults()
    num_done, total = manager.get_final_job_status()
    # if np.sum(manager.final_job_status) < 6:
    if num_done < total:
        # manager.writeFailedObjectives()
        manager.write_status_code(1)
        sys.exit(1)
        logging.info("some job failed, flag as failure")
    else:
        manager.retrieveResults()
        logging.info("successfully retrieved results")
        manager.write_status_code(0)
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        logging.error(f"number of arguments ({len(sys.argv)}) is not 5, exit 1")
        sys.exit(1)

    jobid = sys.argv[1]
    npart = int(sys.argv[2])
    p_eta_point = sys.argv[3]
    particle = sys.argv[4]
    try:
        p_eta_point = json.loads(base64.b64decode(p_eta_point).decode("utf-8"))
        run(jobid, npart, p_eta_point, particle)
    except Exception as ex:
        logging.error(f"failed to run: {ex}")
        logging.error(traceback.format_exc())
        sys.exit(1)
