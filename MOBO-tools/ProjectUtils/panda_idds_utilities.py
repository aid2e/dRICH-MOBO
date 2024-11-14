
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


def get_outcome_value_for_completed_job(job_id):
    objectives = ["piKsep_etalow",
                  "piKsep_etahigh",
                  "acceptance"
                  ]
    # HERE: load results from text file, formatted based on job id
    results = np.loadtxt(os.environ["AIDE_WORKDIR"] + "/log/results/" + "drich-mobo-out_{}.txt".format(job_id))
    if len(objectives) > 1:
        results_dict = {objectives[i]: [results[2 * i], results[2 * i + 1]] for i in range(len(objectives))}
    else:
        results_dict = {objectives[0]: results}
    return results_dict


def run_func(parameters, job_id):
    print(f"start run_func, job_id: {job_id}")

    print("start to create xml")
    create_xml(parameters, job_id)
    print("finished to create xml")

    num_particles = 1500
    num_particles = 1500
    shell_command = ["python3", str(os.environ["AIDE_HOME"]) + "/ProjectUtils/ePICUtils/" + "/runTestsAndObjectiveCalc_local.py", str(job_id), str(num_particles)]
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
        print("failed to run runTestsAndObjectiveCalc_local.py")

    # copy the working directory out for debug
    # print(f"copying directory {os.getcwd()} to /tmp/wguan/test1/")
    # shutil.copytree(os.getcwd(), '/tmp/wguan/test1/', dirs_exist_ok=True)

    ret = get_outcome_value_for_completed_job(job_id)
    print(f"ret: {ret}")

    return ret


init_env = ['source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/setup_mamba.sh;'
            'source /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/wguan/mlcontainer:py311_1.0/opt/conda/dRICH-MOBO/MOBO-tools/setup_new.sh;'
            'export AIDE_HOME=$(pwd);'
            'export PWD_PATH=$(pwd);'
            'export SINGULARITY_OPTIONS="--bind /cvmfs:/cvmfs,$(pwd):$(pwd)"; '
            'env; '
            ]
init_env = " ".join(init_env)


# BNL_OSG_2, BNL_OSG_PanDA_1
@workflow_def(service='panda', source_dir=None, source_dir_parent_level=1, local=True, cloud='US',
              queue='BNL_OSG_2', exclude_source_files=["DTLZ2*", ".*json", ".*log", "work", "log", "OUTDIR", "calibrations", "fieldmaps", "gdml",
                                                       "EICrecon-drich-mobo", "eic-software", "epic-geom-drich-mobo", "irt", "share"],
              return_workflow=True, max_walltime=3600,
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

    """
    def run_multiple(self, trials):
        print("run_multiple")
        return {trial.index: self.run(trial=trial) for trial in trials}
    """

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
            params_list.append([arm.parameters, job_id])
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
        results = {'status': 'running', 'retries': 0, 'result': None}
        return {'results': results}

    def verify_results(self, trial: BaseTrial, results):
        all_arms_finished = True,
        unfinished_arms = []
        print(f"trial {trial.index} work verify_results: {results}")
        for arm in trial.arms:
            job_id = f"{trial.index}_{arm.name}"
            ret = results.get_result(name=None, args=[arm.parameters, job_id])
            if ret is None:
                all_arms_finished = False
                unfinished_arms.append(arm)
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
            job_id = f"{trial.index}_{arm.name}"
            params_list.append([arm.parameters, job_id])

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

        all_finished, all_failed, all_terminated = False, False, False

        status = old_results['status']
        retries = old_results['retries']
        avail_results = old_results['result']
        if status not in ['finished', 'terminated', 'failed']:
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
                        ret = avail_results.get_result(name=None, args=[arm.parameters])
                        if ret is None:
                            print(f"trial {trial.index} work missing results for arm {arm.name}")
                            # get results from new retries
                            if results:
                                new_ret = results.get_result(name=None, args=[arm.parameters])
                                print(f"trial {trial.index} work checking results for arm {arm.name} from new transform {tf_id}, new results: {new_ret}")
                                if new_ret is not None:
                                    print(f"trial {trial.index} work set result for arm {arm.name} from new transform {tf_id} to {new_ret}")
                                    # avail_results.set_result(name=None, args=[arm.parameters], value=new_ret)
                                    old_results['result'].set_result(name=None, args=[arm.parameters], value=new_ret)

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

        trial.update_run_metadata({'results': old_results})

        if all_finished:
            if self.workflow:
                self.workflow.close()
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
            results = results_dict.get('result', None)

            if results is not None:
                for arm in trial.arms:
                    ret = results.get_result(name=None, args=[arm.parameters])
                    if ret is not None:
                        df_dict = {
                            "trial_index": trial.index,
                            "metric_name": self.name,
                            "arm_name": arm.name,
                            "mean": ret.get(self.name, None),
                            "sem": 0.0   # unkown noise
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

            return Ok(value=Data(df=pd.DataFrame.from_records(df_dict_list)))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"trial {trial.index} failed to fetch {self.name}", exception=e)
            )
