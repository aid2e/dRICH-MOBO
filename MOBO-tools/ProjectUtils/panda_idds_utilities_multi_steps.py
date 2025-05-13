
import atexit
import base64
import getpass
import json
import uncertainties
import math
import os
import pandas as pd
import numpy as np
# import shutil
import subprocess
import traceback

from collections import defaultdict
from typing import Iterable

from ax.core.experiment import Experiment

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.runner import Runner
from ax.utils.common.result import Ok, Err

from idds.iworkflow.asyncresult import MapResult
from idds.iworkflow.workflow import workflow as workflow_def      # workflow    # noqa F401
from idds.iworkflow.work import work as work_def

from ProjectUtils.ePICUtils.editxml_local import create_xml


def get_job_status(job_id):
    status_file = os.path.join(os.environ["AIDE_WORKDIR"], "log/results/drich-mobo-status_{}.txt".format(job_id))
    if not os.path.exists(status_file):
        return "running"
    ret_code = -1
    with open(status_file, 'r') as f:
        ret_code = f.read()
        ret_code = int(ret_code)
    if ret_code == 0:
        return "finished"
    else:
        return "failed"


def get_outcome_value_for_completed_job(parameters, job_id, p_eta_point, particle, num_particles):
    # HERE: load results from text file, formatted based on job id
    # results = np.loadtxt(os.environ["AIDE_WORKDIR"] + "/log/results/" + "drich-mobo-out_{}.txt".format(job_id))
    results = np.load(os.environ["AIDE_WORKDIR"] + "/log/results/" + "drich-mobo-out_{}.npz".format(job_id), allow_pickle=True)
    ret = {k: results[k].tolist() for k in results}

    return ret


def run_func_simreco(parameters, job_id, p_eta_point, particle, num_particles=1500, output_file_name=None):
    print(f"start run_func, job_id: {job_id}")

    print("start to create xml")
    create_xml(parameters, job_id)
    print("finished to create xml")

    # num_particles = 1500
    # num_particles = 1500

    if output_file_name:
        output_root_name = output_file_name
    else:
        p = p_eta_point[0]
        eta_min = p_eta_point[1][0]
        eta_max = p_eta_point[1][1]
        output_root_name = "recon_scan_{}_{}_p_{}_eta_{}_{}.root".format(job_id, particle, p, eta_min, eta_max)

    shell_command = ["python3", str(os.environ["AIDE_HOME"]) + "/ProjectUtils/ePICUtils/" + "/runTestsAndObjectiveCalc_local_sep_simreco.py",
                     str(job_id), str(num_particles), base64.b64encode(bytes(json.dumps(p_eta_point), 'ascii')), particle, output_root_name]
    # commandout = subprocess.run(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commandout = subprocess.run(shell_command)
    return_code = commandout.returncode
    # output = commandout.stdout.decode('utf-8')
    # error = commandout.stderr.decode("utf-8")
    output = commandout.stdout
    error = commandout.stderr
    if output:
        output = output.decode('utf-8')
    if error:
        error = error.decode('utf-8')

    print(f"{job_id} run command: {shell_command}")
    print(f"{job_id} return code : {return_code}")
    print(f"============== {job_id} stdout ==============")
    print(output)
    print(f"============== {job_id} end of stdout ==============")
    print(f"============== {job_id} stderr ==============")
    print(error)
    print(f"============== {job_id} end of stderr ==============")

    if return_code != 0:
        print("failed to run runTestsAndObjectiveCalc_local_sep.py")

    # copy the working directory out for debug
    # print(f"copying directory {os.getcwd()} to /tmp/wguan/test1/")
    # shutil.copytree(os.getcwd(), '/tmp/wguan/test1/', dirs_exist_ok=True)

    # ret = get_outcome_value_for_completed_job(parameters, job_id, p_eta_point, particle, num_particles)
    # print(f"ret: {ret}")

    return return_code


def run_func_analy(parameters, job_id, p_eta_point, particle, num_particles=1500, input_file_name=None):
    print(f"start run_func, job_id: {job_id}")

    print("start to create xml")
    create_xml(parameters, job_id)
    print("finished to create xml")

    # num_particles = 1500
    # num_particles = 1500

    if input_file_name:
        if isinstance(input_file_name, (list, tuple)):
            input_root_name = input_file_name[0]
        else:
            input_root_name = input_file_name
    else:
        p = p_eta_point[0]
        eta_min = p_eta_point[1][0]
        eta_max = p_eta_point[1][1]
        input_root_name = "recon_scan_{}_{}_p_{}_eta_{}_{}.root".format(job_id, particle, p, eta_min, eta_max)

    shell_command = ["python3", str(os.environ["AIDE_HOME"]) + "/ProjectUtils/ePICUtils/" + "/runTestsAndObjectiveCalc_local_sep_analy.py",
                     str(job_id), str(num_particles), base64.b64encode(bytes(json.dumps(p_eta_point), 'ascii')), particle, input_root_name]
    # commandout = subprocess.run(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commandout = subprocess.run(shell_command)
    return_code = commandout.returncode
    # output = commandout.stdout.decode('utf-8')
    # error = commandout.stderr.decode("utf-8")
    output = commandout.stdout
    error = commandout.stderr
    if output:
        output = output.decode('utf-8')
    if error:
        error = error.decode('utf-8')

    print(f"{job_id} run command: {shell_command}")
    print(f"{job_id} return code : {return_code}")
    print(f"============== {job_id} stdout ==============")
    print(output)
    print(f"============== {job_id} end of stdout ==============")
    print(f"============== {job_id} stderr ==============")
    print(error)
    print(f"============== {job_id} end of stderr ==============")

    if return_code != 0:
        print("failed to run runTestsAndObjectiveCalc_local_sep.py")

    # copy the working directory out for debug
    # print(f"copying directory {os.getcwd()} to /tmp/wguan/test1/")
    # shutil.copytree(os.getcwd(), '/tmp/wguan/test1/', dirs_exist_ok=True)

    ret = get_outcome_value_for_completed_job(parameters, job_id, p_eta_point, particle, num_particles)
    print(f"ret: {ret}")

    return ret


def harmonic_mean(values, weights):
    denom = 0
    if np.any(values == 0):
        return uncertainties.ufloat(0, 0)
    for (weight, value) in zip(weights, values):
        denom += weight / value
    return np.sum(weights) / (denom)


def get_final_result(name, p_eta_points, particles, n_particles, n_particles_per_job, trial, result):
    final_results = {}
    print(f"trial {trial.index} get_final_result {result}")
    print(f"trial {trial.index} particles {particles} n_particles {n_particles} n_particles_per_job {n_particles_per_job} p_eta_points {p_eta_points}")
    for arm in trial.arms:

        job_id = f"{trial.index}_{arm.name}"

        results_nsigma = []
        results_eff = []
        result_p = []
        result_etalow = []

        for i, p_eta_point in enumerate(p_eta_points):
            particles_ret = {}
            plus_cher = []
            for particle in particles:
                # [arm.parameters, job_id, p_eta_point, particle, n_particles]
                params = get_job_param(name, p_eta_point, particle, n_particles, n_particles_per_job, trial, arm, job_id)
                job_params, output_job_name, output_dataset_name, output_root_name, input_job_name, input_dataset_name, output_log_dataset_name, job_key = params

                ret = result.get_result(name=name, key=job_key, args=job_params, verbose=True)
                particles_ret[particle] = ret
                plus_cher.append(ret['0'].get('plus_cher', None))
            print(f"i {i} p_eta_point {p_eta_point} particles_ret {particles_ret} plus_cher {plus_cher}")

            p = p_eta_point[0]
            eta_min = p_eta_point[1][0]
            # eta_max = p_eta_point[1][1]

            mean_nphot = (plus_cher[0][0] + plus_cher[1][0]) / 2
            mean_sigma = (plus_cher[0][2] + plus_cher[1][2]) / 2
            # calculate nsigma separation
            if mean_sigma != 0:
                nsigma = (abs(plus_cher[0][1] - plus_cher[1][1]) * math.sqrt(mean_nphot)) / mean_sigma
            else:
                nsigma = 0
            # get mean fraction of tracks with reco photons
            mean_eff = (plus_cher[0][3] + plus_cher[1][3]) / 2

            results_nsigma.append(uncertainties.ufloat(nsigma, p_eta_point[3] * nsigma))
            results_eff.append(uncertainties.ufloat(mean_eff, p_eta_point[4] * mean_eff))
            result_p.append(p)
            result_etalow.append(eta_min)

        result_p = np.array(result_p)
        result_etalow = np.array(result_etalow)
        results_nsigma = np.array(results_nsigma)
        results_eff = np.array(results_eff)

        print(f"trial {trial.index} arm {arm.name} get_final_result: result_p {result_p}")
        print(f"trial {trial.index} arm {arm.name} get_final_result: result_etalow {result_etalow}")
        print(f"trial {trial.index} arm {arm.name} get_final_result: results_nsigma {results_nsigma}")
        print(f"trial {trial.index} arm {arm.name} get_final_result: results_eff {results_eff}")

        # average together low/mid and mid/high bins (reduce dimensions but still have some info
        # from mid \eta)
        sep_etalow = harmonic_mean(np.array([results_nsigma[0], results_nsigma[3]]), np.array([1., 1.]))
        sep_etamid = harmonic_mean(np.array([results_nsigma[1], results_nsigma[4]]), np.array([1., 1.]))
        sep_etahigh = harmonic_mean(np.array([results_nsigma[2], results_nsigma[5]]), np.array([1., 1.]))

        mean_sep_low = harmonic_mean(np.array([sep_etalow, sep_etamid]), np.array([1.0, 0.5]))
        mean_sep_high = harmonic_mean(np.array([sep_etamid, sep_etahigh]), np.array([0.5, 1.0]))

        acc_all = np.mean(results_eff)

        final_result = np.array([mean_sep_low.n, mean_sep_low.s,
                                 mean_sep_high.n, mean_sep_high.s,
                                 acc_all.n, acc_all.s])
        final_result = {
            "piKsep_etalow": [mean_sep_low.n, mean_sep_low.s],
            "piKsep_etahigh": [mean_sep_high.n, mean_sep_high.s],
            "acceptance": [acc_all.n, acc_all.s]
        }

        final_results[arm.name] = final_result
    print(f"trial {trial.index} arm {arm.name} get_final_result: final results: {final_results}")
    return final_results


# 'source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/setup_mamba.sh;'
# 'source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/dRICH-MOBO/MOBO-tools/setup_new.sh;'

init_env = ['source /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fyingtsai/eic_xl:24.11.1/opt/conda/setup_mamba.sh;'
            'source /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fyingtsai/eic_xl:24.11.1/opt/conda/dRICH-MOBO//MOBO-tools/setup_new.sh;'
            'command -v singularity &> /dev/null || export SINGULARITY=/cvmfs/oasis.opensciencegrid.org/mis/singularity/current/bin/singularity;'
            'export AIDE_HOME=$(pwd);'
            'export PWD_PATH=$(pwd);'
            'export SINGULARITY_OPTIONS="--bind /cvmfs:/cvmfs,$(pwd):$(pwd)"; '
            'export SIF=/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:24.11.1-stable; export SINGULARITY_BINDPATH=/cvmfs,/afs; '
            'env; '
            ]
init_env = " ".join(init_env)


# BNL_OSG_2, BNL_OSG_PanDA_1, BNL_PanDA_1
@workflow_def(service='panda', source_dir=None, source_dir_parent_level=1, local=True, cloud='US',
              queue='BNL_PanDA_1', exclude_source_files=["DTLZ2*", ".*json", ".*log", "work", "log", "OUTDIR", "calibrations", "fieldmaps", "gdml",
                                                             "EICrecon-drich-mobo", "eic-software", "epic-geom-drich-mobo", "irt", "share", "back*"],
              return_workflow=True, max_walltime=3600,
              core_count=1, total_memory=4000,   # MB
              init_env=init_env, enable_separate_log=True
              # container_options={'container_image': '/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:24.11.1-stable'}
              # container_options={'container_image': '/cvmfs/unpacked.cern.ch/registry.hub.docker.com/fyingtsai/eic_xl:24.11.1'}
              # container_options={'container_image': '/cvmfs/singularity.opensciencegrid.org/htc/rocky:9'}
              )
def empty_workflow_func():
    pass


def build_experiment_pandaidds(search_space, optimization_config, runner):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=runner
    )
    return experiment


def get_user_name():
    rucio_account = os.environ.get('RUCIO_ACCOUNT', None)
    if rucio_account:
        return rucio_account

    username = getpass.getuser()
    return username


def get_job_param(name, p_eta_point, particle, n_particles, n_particles_per_job, trial, arm, job_id):
    p = p_eta_point[0]
    eta_min = p_eta_point[1][0]
    eta_max = p_eta_point[1][1]

    job_key = f'{job_id}_{particle}_{p}_{eta_min}_{eta_max}'

    output_job_name = f'step1_simreco_run_func_{job_id}_{particle}_{p}_{eta_min}_{eta_max}'
    username = get_user_name()
    output_dataset_name = f'user.{username}.{name}.{output_job_name}.$WORKFLOWID/'
    # output_log_dataset_name = f'user.{username}.{name}.{output_job_name}.$WORKFLOWID.log/'
    input_dataset_name = f'user.{username}:user.{username}.{name}.{output_job_name}.$WORKFLOWID/'
    output_root_name = "recon_scan_{}_{}_p_{}_eta_{}_{}.root".format(job_id, particle, p, eta_min, eta_max)
    output_dataset_name = output_dataset_name.replace("+", "_")     # rucio dataset name schema requirement
    output_root_name = output_root_name.replace("+", "_")
    input_dataset_name = input_dataset_name.replace("+", "_")

    input_job_name = f'step1_analy_run_func_{job_id}_{particle}_{p}_{eta_min}_{eta_max}'
    output_log_dataset_name = f'user.{username}.{name}.{input_job_name}.$WORKFLOWID.log/'
    output_log_dataset_name = output_log_dataset_name.replace("+", "_")

    # job_params = [arm.parameters, job_id, p_eta_point, particle, n_particles, n_particles_per_job, output_root_name]
    job_params = [arm.parameters, job_id, p_eta_point, particle, n_particles_per_job, output_root_name]
    return job_params, output_job_name, output_dataset_name, output_root_name, input_job_name, input_dataset_name, output_log_dataset_name, job_key


class PanDAIDDSJobRunner(Runner):
    def __init__(self, name='pandaidds'):
        self._runner_funcs = {'step1_simreco_function': run_func_simreco, 'step2_analy_function': run_func_analy, 'pre_kwargs': None}
        self.transforms = {}
        self.workflow = None
        self.retries = 0

        self.name = name

        # self.workflow = empty_workflow_func()
        # print(self.workflow)
        # self.workflow.pre_run()

        self.failed_result = -1

        self.n_particles = 2000
        self.n_particles_per_job = 1000

        self.particles = ["pi+", "kaon+"]
        self.p_eta_points = [
            [15, [1.5, 2.0], 0, 0.02457205, 0.00878185],
            [15, [2.0, 2.5], 0, 0.01871374, 0.00916126],
            [15, [2.5, 3.5], 0, 0.02046443, 0.01240257],
            [40, [1.5, 2.0], 1, 0.04405797, 0.00824205],
            [40, [2.0, 2.5], 1, 0.01390604, 0.00350744],
            [40, [2.5, 3.5], 1, 0.01391206, 0.00349814]
        ]

        self.n_particles = 200
        self.n_particles_per_job = 100

        """
        self.particles = ["pi+", "kaon+"]
        self.p_eta_points = [
            [15, [1.5, 2.0], 0, 0.02457205, 0.00878185]
        ]
        """

        atexit.register(self.cleanup)

    def cleanup(self):
        if self.workflow:
            self.workflow.close()

    def run(self, trial: BaseTrial):
        try:
            ret = self.run_local(trial)
            print(f"run trail {trial.index} result: {ret}")
            return ret
        except Exception as ex:
            print(f"PanDAIDDSJobRunner run exception: {ex} {traceback.format_exc()}")
        except:
            print("PanDAIDDSJobRunner run exception")

    def run_local(self, trial: BaseTrial):
        to_submit_workflow = False
        if self.workflow is None:
            print("Define workflow")
            to_submit_workflow = True
            self.workflow = empty_workflow_func()
            print(self.workflow)
            self.workflow.pre_run()

        if not isinstance(trial, BaseTrial):
            raise ValueError("This runner only handles `BaseTrial`.")

        params_list = []
        print(f"run trial {trial.index} num of arms: {len(trial.arms)}")
        print(f"run trial {trial.index} arms: {trial.arms}")

        for arm in trial.arms:
            job_id = f"{trial.index}_{arm.name}"

            for p_eta_point in self.p_eta_points:
                for particle in self.particles:
                    # test
                    params = get_job_param(self.name, p_eta_point, particle, self.n_particles, self.n_particles_per_job, trial, arm, job_id)
                    # job_params, output_job_name, output_dataset_name, output_root_name, input_job_name = params
                    params_list.append(params)
            # params_list.append([arm.parameters, job_id])
        # print(params_list)

        self.transforms[trial.index] = {}
        # self.transforms[trial.index][self.retries] = {}

        # one work is one objective
        # with multiple objectives, there will be multiple work objects

        step1_simreco_function = self._runner_funcs['step1_simreco_function']
        step2_analy_function = self._runner_funcs['step2_analy_function']
        pre_kwargs = self._runner_funcs['pre_kwargs']

        # prepare workflow is after the work.store.
        # in this way, the workflow will upload the work's files
        # self.workflow.store()
        if to_submit_workflow:
            print("prepare workflow")
            self.workflow.prepare()
            print("submit workflow")
            req_id = self.workflow.submit()
            print(f"workflow id: {req_id}")

        for params in params_list:
            job_params, output_job_name, output_dataset_name, output_root_name, input_job_name, input_dataset_name, output_log_dataset_name, job_key = params

            step1_work = work_def(step1_simreco_function, workflow=self.workflow, return_work=True, pre_kwargs=pre_kwargs, map_results=True,
                                  name=output_job_name, output_file_name=output_root_name, output_dataset_name=output_dataset_name,
                                  num_events=self.n_particles, num_events_per_job=self.n_particles_per_job)
            # step1_w = step1_work(multi_jobs_kwargs_list=params_list)
            # step1_w = step1_work(args=job_params)
            step1_w = step1_work(*job_params)
            # w.store()
            # print(f"trial {trial.index}: create a task: ({w.internal_id}) {step1_w}")
            # self.transforms[trial.index][self.retries] = {'tf_id': None, 'work': step1_w, 'results': None, 'status': 'new'}

            print("submit step1 work")
            step1_tf_id = step1_w.submit()
            print(f"trial {trial.index} step1 work {step1_w.internal_id} transform id: {step1_tf_id}")
            if not step1_tf_id:
                raise Exception("Failed to submit work to PanDA")

            self.transforms[trial.index][input_job_name] = {}

            # job_id = job_params[1]
            # particle = job_params[3]
            # job_key = f'{job_id}_{particle}'  # make sure the job_key is unique per work, different key can be used.
            step2_work = work_def(step2_analy_function, workflow=self.workflow, return_work=True, pre_kwargs=pre_kwargs, map_results=True,
                                  name=input_job_name, input_datasets={'input_file_name': input_dataset_name}, parent_transform_id=step1_tf_id,
                                  log_dataset_name=output_log_dataset_name, parent_internal_id=step1_w.internal_id, job_key=job_key)
            # step2_w = step2_work(multi_jobs_kwargs_list=params_list)
            # step2_w = step2_work(args=job_params)
            # job_params = [arm.parameters, job_id, p_eta_point, particle, n_particles_per_job, output_root_name]
            job_params_without_input = job_params[:-1]   # the last arg 'input_file_name' will be filled by input_datasets
            step2_w = step2_work(*job_params_without_input)
            print(f"trial {trial.index}: create a task: ({step2_w.internal_id}) {step2_w}")
            self.transforms[trial.index][input_job_name][self.retries] = {'tf_id': None, 'work': step2_w, 'results': None, 'status': 'new'}

            print("submit step1 work")
            step2_tf_id = step2_w.submit()
            print(f"trial {trial.index} step2 work {step2_w.internal_id} {input_job_name} transform id: {step2_tf_id}")
            if not step2_tf_id:
                raise Exception("Failed to submit work to PanDA")

            step2_w.init_async_result()

            # arm_parameters, job_id, p_eta_point, particle, n_particles_per_job, output_root_name = job_params
            # job_params[-1] = "$WORKINPUTS1"       # set a placeholder as the inputs (output of the previous job)
            # job_params_without_input = job_params[:-1]   # input_file_name
            self.transforms[trial.index][input_job_name][self.retries] = {'tf_id': step2_tf_id, 'work': step2_w,
                                                                          'results': None, 'status': 'new',
                                                                          'job_key': job_key,
                                                                          'job_params': job_params_without_input}
        results = {'status': 'running', 'retries': 0, 'result': None, 'combine_result': None}
        return {'results': results}

    def get_trial_status(self, trial: BaseTrial):
        ret_results = trial.run_metadata.get("results")

        tmp_map_result = MapResult()
        if not ret_results['result']:
            ret_results['result'] = {}
        # MapResult cannot be json serializable. So we only use it as a temp.
        # When saving to Ax, we convert it to/from dict.
        tmp_map_result.set_from_dict_results(ret_results['result'])

        all_terminated = True
        has_failures = False

        # print(f"transforms: {self.transforms}")
        for obj in self.transforms[trial.index]:
            for retries in self.transforms[trial.index][obj]:
                w = self.transforms[trial.index][obj][retries]['work']
                # tf_if = self.transforms[trial.index][obj][retries]['tf_id']
                w.init_async_result()

                print(f"trial {trial.index} {obj} work {w.internal_id} retries {retries} status: {w.get_status()}")
                if not w.is_terminated():
                    all_terminated = False
                else:
                    results = w.get_results()

                    print(f"trial {trial.index} {obj} work {w.internal_id} retries {retries} results: {results}")
                    self.transforms[trial.index][obj][retries]['results'] = results
                    if w.is_finished():
                        print(f"trial {trial.index} {obj} work {w.internal_id} retries {retries} finished")
                    elif w.is_failed():
                        has_failures = True
                        print(f"trial {trial.index} {obj} work {w.internal_id} retries {retries} failed")
                    else:
                        print(f"trial {trial.index} {obj} work {w.internal_id} retries {retries} terminated")

                    job_params = self.transforms[trial.index][obj][retries]['job_params']
                    job_id = job_params[1]
                    job_key = self.transforms[trial.index][obj][retries]['job_key']
                    ret = results.get_result(name=self.name, args=job_params, key=job_key, verbose=True)
                    print(f"trial {job_id} job_params {job_params} key {job_key} ret: {ret}")
                    tmp_map_result.set_result(name=self.name, key=job_key, args=job_params, value=ret)

        ret_status = TrialStatus.RUNNING
        if all_terminated:
            combine_results = get_final_result(self.name, self.p_eta_points, self.particles, self.n_particles,
                                               self.n_particles_per_job, trial, tmp_map_result)
            ret_results['combine_result'] = combine_results

            if has_failures:
                ret_results['status'] = 'failed'
                ret_status = TrialStatus.FAILED
            else:
                ret_status = TrialStatus.COMPLETED
                ret_results['status'] = 'finished'

        ret_results['result'] = tmp_map_result.get_dict_results()  # convert MapResult to dict
        trial.update_run_metadata({'results': ret_results})

        return ret_status

    def poll_trial_status(self, trials: Iterable[BaseTrial]):
        # print("poll_trial_status")
        status_dict = defaultdict(set)
        for trial in trials:
            status = self.get_trial_status(trial)
            print("poll_trial_status trail %s: status: %s" % (trial.index, status))
            status_dict[status].add(trial.index)

            try:
                # trial.mark_as(status)
                pass
            except Exception:
                pass
            except ValueError:
                pass
        # print(status_dict)
        return status_dict


class PanDAIDDSJobMetric(Metric):  # Pulls data for trial from external system.
    # def __init__(self, name: str, lower_is_better: Optional[bool] = None, properties: Optional[Dict[str, Any]] = None, function: optional[Any] = None) -> None:

    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        print(f"trial {trial.index} fetch_trial_data")
        if not isinstance(trial, BaseTrial):
            raise ValueError("This metric only handles `BaseTrial`.")

        try:
            results_dict = trial.run_metadata.get("results")
            df_dict_list = []
            results = results_dict.get('combine_result', None)
            print(f"trial {trial.index} fetch_trial_data results: {results}")

            if results is not None:
                for arm in trial.arms:
                    ret = results.get(arm.name, None)
                    ret_metric = None
                    if ret is not None:
                        ret_metric = ret.get(self.name, None)

                    if ret_metric:
                        print(f"trial {trial.index} {self.name} result for {arm.name}: {ret_metric}")
                        df_dict = {
                            "trial_index": trial.index,
                            "metric_name": self.name,
                            "arm_name": arm.name,
                            "mean": ret_metric[0],
                            "sem": ret_metric[1]
                        }
                        df_dict_list.append(df_dict)
                    else:
                        print(f"trial {trial.index} {self.name} misses result for {arm.name}")
                        df_dict = {
                            "trial_index": trial.index,
                            "metric_name": self.name,
                            "arm_name": arm.name,
                            "mean": ret,
                            "sem": 0.0   # unkown noise
                        }
                        df_dict_list.append(df_dict)

            print(f"fetch_trial_data metric {self.name} df_dict_list: {df_dict_list}")
            return Ok(value=Data(df=pd.DataFrame.from_records(df_dict_list)))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"trial {trial.index} failed to fetch {self.name}", exception=e)
            )
