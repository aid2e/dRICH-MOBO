import os
import pandas as pd
import numpy as np
import subprocess

from collections import defaultdict
from typing import Iterable

from ax.core.experiment import Experiment

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.runner import Runner
from ax.utils.common.result import Ok, Err

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
    create_xml(parameters, job_id)
    num_particles = 2
    shell_command = ["python3", str(os.environ["AIDE_HOME"]) + "/ProjectUtils/ePICUtils/" + "/runTestsAndObjectiveCalc_local.py", str(job_id), str(num_particles)]
    # commandout = subprocess.run(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commandout = subprocess.run(shell_command)
    return_code = commandout.returncode
    # Handle special exit code for geometry overlaps
    if return_code == 42:
        print(f"[ERROR] Job {job_id} failed due to geometry overlaps — marking trial as FAILED.")
        return {
            "piKsep_etahigh": [0.0, 0.0],
            "piKsep_etalow": [0.0, 0.0],
            "acceptance": [0.0, 0.0],
            "status": "FAILED"
        }
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

    ret = get_outcome_value_for_completed_job(job_id)
    print(f"ret: {ret}")
    return ret


def build_experiment_local(search_space, optimization_config, runner):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=runner
    )
    return experiment


class LocalJobRunner(Runner):
    def __init__(self):
        pass

    def run(self, trial: BaseTrial):
        metadata= {} 
        for arm in trial.arms:
            job_id = f"{trial.index}_{arm.name}"
            result = run_func(arm.parameters, job_id)
            #check if run_func returned a failur flag
            if isinstance(result, dict) and result.get("status") == "FAILED":
                raise RuntimeError(f"[Ax] Trial {trial.index} failed due to geometry overlaps.")
            metadata[arm.name]=result
        return metadata

    def poll_trial_status(self, trials: Iterable[BaseTrial]):
        # print("poll_trial_status")
        status_dict = defaultdict(set)
        for trial in trials:
            status_job_id = {}
            for arm in trial.arms:
                job_id = f"{trial.index}_{arm.name}"
                status = get_job_status(job_id)
                if status not in status_job_id:
                    status_job_id[status] = []
                status_job_id[status].append(job_id)

            keys = list(status_job_id.keys())
            if 'running' in keys:
                status = TrialStatus.RUNNING
            else:
                if len(keys) == 1 and keys[0] == 'finished':
                    status = TrialStatus.COMPLETED
                else:
                    status = TrialStatus.FAILED
            print("trail %s: status: %s" % (trial.index, status))
            status_dict[status].add(trial.index)
            try:
                trial.mark_as(status)
            except Exception:
                pass
            except ValueError:
                pass
        # print(status_dict)
        return status_dict


class LocalJobMetric(Metric):  # Pulls data for trial from external system.
    # def __init__(self, name: str, lower_is_better: Optional[bool] = None, properties: Optional[Dict[str, Any]] = None, function: optional[Any] = None) -> None:

    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        print(f"trial {trial.index} fetch_trial_data")
        if not isinstance(trial, BaseTrial):
            raise ValueError("This metric only handles `BaseTrial`.")

        try:
            df_dict_list = []
            for arm in trial.arms:
                job_id = f"{trial.index}_{arm.name}"
                ret_dict = get_outcome_value_for_completed_job(job_id)

                df_dict = {
                    "trial_index": trial.index,
                    "metric_name": self.name,
                    "arm_name": arm.name,
                    "mean": ret_dict.get(self.name)[0],
                    "sem": ret_dict.get(self.name)[1]
                }
                df_dict_list.append(df_dict)

            return Ok(value=Data(df=pd.DataFrame.from_records(df_dict_list)))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"trial {trial.index} failed to fetch {self.name}", exception=e)
            )
