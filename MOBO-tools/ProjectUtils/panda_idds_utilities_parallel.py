
import atexit
import base64
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


def run_func(parameters, job_id, p_eta_point, particle, num_particles=1500):
    print(f"start run_func, job_id: {job_id}")

    print("start to create xml")
    create_xml(parameters, job_id)
    print("finished to create xml")

    # num_particles = 1500
    # num_particles = 1500
    shell_command = ["python3", str(os.environ["AIDE_HOME"]) + "/ProjectUtils/ePICUtils/" + "/runTestsAndObjectiveCalc_local_sep.py",
                     str(job_id), str(num_particles), base64.b64encode(bytes(json.dumps(p_eta_point), 'ascii')), particle]
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


def get_final_result(p_eta_points, particles, n_particles, trial, result):
    final_results = {}
    print(f"trial {trial.index} get_final_result")
    print(f"trial {trial.index} particles {particles} n_particles {n_particles} p_eta_points {p_eta_points}")
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
                # [arm.parameters, job_id, p_eta_point, particle, self.n_particles]
                ret = result.get_result(name=None, args=[arm.parameters, job_id, p_eta_point, particle, n_particles])
                particles_ret[particle] = ret
                plus_cher.append(ret['0'].get('plus_cher', None))

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


init_env = ['source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/setup_mamba.sh;'
            'source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/dRICH-MOBO/MOBO-tools/setup_new.sh;'
            'command -v singularity &> /dev/null || export SINGULARITY=/cvmfs/oasis.opensciencegrid.org/mis/singularity/current/bin/singularity;'
            'export AIDE_HOME=$(pwd);'
            'export PWD_PATH=$(pwd);'
            'export SINGULARITY_OPTIONS="--bind /cvmfs:/cvmfs,$(pwd):$(pwd)"; '
            'env; '
            ]
init_env = " ".join(init_env)


# BNL_OSG_2, BNL_OSG_PanDA_1
@workflow_def(service='panda', source_dir=None, source_dir_parent_level=1, local=True, cloud='US',
              queue='BNL_OSG_PanDA_1', exclude_source_files=["DTLZ2*", ".*json", ".*log", "work", "log", "OUTDIR", "calibrations", "fieldmaps", "gdml",
                                                       "EICrecon-drich-mobo", "eic-software", "epic-geom-drich-mobo", "irt", "share"],
              return_workflow=True, max_walltime=3600,
              core_count=1, total_memory=4000,   # MB
              init_env=init_env,
              container_options={'container_image': '/cvmfs/singularity.opensciencegrid.org/eicweb/jug_xl:24.08.1-stable'})
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


class PanDAIDDSJobRunner(Runner):
    def __init__(self):
        self._runner_funcs = {'function': run_func, 'pre_kwargs': None}
        self.transforms = {}
        self.workflow = None
        self.retries = 0

        # self.workflow = empty_workflow_func()
        # print(self.workflow)
        # self.workflow.pre_run()

        self.n_particles = 1500
        self.particles = ["pi+", "kaon+"]
        self.p_eta_points = [
            [15, [1.5, 2.0], 0, 0.02457205, 0.00878185],
            [15, [2.0, 2.5], 0, 0.01871374, 0.00916126],
            [15, [2.5, 3.5], 0, 0.02046443, 0.01240257],
            [40, [1.5, 2.0], 1, 0.04405797, 0.00824205],
            [40, [2.0, 2.5], 1, 0.01390604, 0.00350744],
            [40, [2.5, 3.5], 1, 0.01391206, 0.00349814]
        ]

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
                    params_list.append([arm.parameters, job_id, p_eta_point, particle, self.n_particles])
            # params_list.append([arm.parameters, job_id])
        # print(params_list)

        self.transforms[trial.index] = {}
        self.transforms[trial.index][self.retries] = {}

        # one work is one objective
        # with multiple objectives, there will be multiple work objects

        function = self._runner_funcs['function']
        pre_kwargs = self._runner_funcs['pre_kwargs']

        work = work_def(function, workflow=self.workflow, return_work=True, pre_kwargs=pre_kwargs, map_results=True, name=f'run_func_{trial.index}')
        w = work(multi_jobs_kwargs_list=params_list)
        # w.store()
        print(f"trial {trial.index}: create a task: ({w.internal_id}) {w}")
        self.transforms[trial.index][self.retries] = {'tf_id': None, 'work': w, 'results': None, 'status': 'new'}

        # prepare workflow is after the work.store.
        # in this way, the workflow will upload the work's files
        # self.workflow.store()
        if to_submit_workflow:
            print("prepare workflow")
            self.workflow.prepare()
            print("submit workflow")
            req_id = self.workflow.submit()
            print(f"workflow id: {req_id}")

        print("submit work")
        tf_id = w.submit()
        print(f"trial {trial.index} work {w.internal_id} transform id: {tf_id}")
        if not tf_id:
            raise Exception("Failed to submit work to PanDA")
        w.init_async_result()
        self.transforms[trial.index][self.retries] = {'tf_id': tf_id, 'work': w, 'results': None, 'status': 'new'}
        results = {'status': 'running', 'retries': 0, 'result': None, 'combine_result': None}
        return {'results': results}

    def verify_results(self, trial: BaseTrial, results):
        all_arms_finished = True,
        unfinished_arms = []
        print(f"trial {trial.index} work verify_results: {results}")
        for arm in trial.arms:
            job_id = f"{trial.index}_{arm.name}"

            for p_eta_point in self.p_eta_points:
                for particle in self.particles:
                    # test
                    ret = results.get_result(name=None, args=[arm.parameters, job_id, p_eta_point, particle, self.n_particles])
                    if ret is None:
                        all_arms_finished = False
                        unfinished_arms.append([arm.parameters, job_id, p_eta_point, particle, self.n_particles])
        return all_arms_finished, unfinished_arms

    def submit_retries(self, trial, retries, unfinished_arms):
        print(f"trial {trial.index} work submit retries {retries} for {unfinished_arms}")
        self.transforms[trial.index][retries] = {}

        # one work is one objective
        # with multiple objectives, there will be multiple work objects

        function = self._runner_funcs['function']
        pre_kwargs = self._runner_funcs['pre_kwargs']
        params_list = []
        for arm in unfinished_arms:
            # job_id = f"{trial.index}_{arm.name}"
            params_list.append(arm)

        work = work_def(function, workflow=self.workflow, return_work=True, pre_kwargs=pre_kwargs, map_results=True, name=f'run_func_{trial.index}')

        w = work(multi_jobs_kwargs_list=params_list)
        # w.store()
        print(f"trial {trial.index}, retries {retries}: create a task: {w}")
        tf_id = w.submit()
        print(f"trial {trial.index}, retries {retries}: submit a task: {w}, {tf_id}")
        if not tf_id:
            raise Exception("Failed to submit work to PanDA")
        self.transforms[trial.index][retries] = {'tf_id': tf_id, 'work': w, 'results': None}

    def get_trial_status(self, trial: BaseTrial):
        old_results = trial.run_metadata.get("results")

        if old_results['result']:
            # convert dict to iDDS MapResult
            map_result = MapResult()
            old_results['result'] = map_result.set_from_dict_results(old_results['result'])

        all_finished, all_failed, all_terminated = False, False, False

        status = old_results['status']
        retries = old_results['retries']
        avail_results = old_results['result']
        # if status not in ['finished', 'terminated', 'failed']:
        if True:
            w = self.transforms[trial.index][retries]['work']
            tf_id = self.transforms[trial.index][retries]['tf_id']
            w.init_async_result()

            # results = w.get_results()
            # print(results)

            if w.is_terminated():
                results = w.get_results()
                print(f"trial {trial.index} work retries {retries} results: {results}")
                self.transforms[trial.index][retries]['results'] = results
                if w.is_finished():
                    print(f"trial {trial.index} work retries {retries} finished")
                    old_results['status'] = 'finished'
                elif w.is_failed():
                    print(f"trial {trial.index} work retries {retries} failed")
                    old_results['status'] = 'failed'
                    all_failed = True
                else:
                    print(f"trial {trial.index} work retries {retries} terminated")
                    old_results['status'] = 'terminated'
                    all_terminated = True

                if avail_results is None:
                    old_results['result'] = results
                else:
                    for arm in trial.arms:
                        job_id = f"{trial.index}_{arm.name}"
                        for p_eta_point in self.p_eta_points:
                            for particle in self.particles:
                                ret = avail_results.get_result(name=None, args=[arm.parameters, job_id, p_eta_point, particle, self.n_particles])
                                if ret is None:
                                    job_par = f"job_id {job_id} p_eta_point {p_eta_point} particle {particle} self.n_particles {self.n_particles}"
                                    print(f"trial {trial.index} work missing results for arm {arm.name} {job_par}")
                                    # get results from new retries
                                    if results:
                                        new_ret = results.get_result(name=None, args=[arm.parameters, job_id, p_eta_point, particle, self.n_particles])
                                        print(f"trial {trial.index} work checking results for arm {arm.name} {job_par} from new transform {tf_id}, new results: {new_ret}")
                                        if new_ret is not None:
                                            print(f"trial {trial.index} work set result for arm {arm.name} {job_par} from new transform {tf_id} to {new_ret}")
                                            # avail_results.set_result(name=None, args=[arm.parameters], value=new_ret)
                                            old_results['result'].set_result(name=None, args=[arm.parameters, job_id, p_eta_point, particle, self.n_particles], value=new_ret)

                ret, unfinished_arms = self.verify_results(trial, old_results['result'])
                print(f"trial {trial.index} work verify_results: ret: {ret}, unfinished_arms: {unfinished_arms}")
                if unfinished_arms:
                    all_finished = False
                else:
                    all_finished = True
                if unfinished_arms and retries < 3:
                    retries += 1
                    all_finished, all_failed, all_terminated = False, False, False
                    self.submit_retries(trial, retries, unfinished_arms)
                    old_results['retries'] = retries
                    old_results['status'] = 'running'
                    all_failed, all_terminated = False, False

        if all_finished:
            print(f"all_finished: {all_finished}, get the final combined result.")
            old_results['combine_result'] = get_final_result(self.p_eta_points, self.particles, self.n_particles, trial, old_results['result'])

        if old_results['result']:
            # to convert iDDS MapResult to dict, which can be serialized with json
            old_results['result'] = old_results['result'].get_dict_results()

        trial.update_run_metadata({'results': old_results})

        if all_finished:
            return TrialStatus.COMPLETED
        elif all_failed:
            return TrialStatus.FAILED
        elif all_terminated:
            # no subfinished status
            return TrialStatus.COMPLETED

        return TrialStatus.RUNNING

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
