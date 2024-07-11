import pandas as pd

from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.utils.common.result import Ok, Err
from ax.core.trial import Trial
from .slurm_utilities import get_slurm_queue_client

class SlurmJobMetric(Metric):  # Pulls data for trial from external system.
    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:

        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")

        try:
            slurm_job_queue = get_slurm_queue_client()

            metric_data = slurm_job_queue.get_outcome_value_for_completed_job(
                trial.run_metadata.get("job_id")
            )
            #TODO: get real uncertainty/noise estimate
            sem = 0.0
            df_dict = {
                "trial_index": trial.index,
                "metric_name": self.name,
                "arm_name": trial.arm.name,
                "mean": metric_data.get(self.name),
                "sem": sem
            }
            return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )
