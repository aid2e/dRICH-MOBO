from collections import defaultdict
from typing import Iterable, Set

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.core.trial import Trial
from .slurm_utilities import get_slurm_queue_client

class SlurmJobRunner(Runner):  # Deploys trials to external system.
    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # list of objectives (to pass to SlurmQueueClient)
        self.metrics = metrics
    def run(self, trial: BaseTrial):
        if not isinstance(trial, BaseTrial):
            raise ValueError("This runner only handles `BaseTrial`.")

        slurm_job_queue = get_slurm_queue_client()
        # supply objective names if not already set for SlurmQueueClient
        if slurm_job_queue.metrics == None:
            slurm_job_queue.metrics = self.metrics
            
        return_job_id = []
        for arm in trial.arms:
            job_id = slurm_job_queue.schedule_job_with_parameters(
                parameters=arm.parameters
            )
            return_job_id.append(job_id)
        if len(return_job_id)==1:
            return {"job_id": return_job_id[0]}
        else:
            return {"job_id": return_job_id}
    def poll_trial_status(self, trials: Iterable[BaseTrial]):
        status_dict = defaultdict(set)
        for trial in trials:
            slurm_job_queue = get_slurm_queue_client()
            status = slurm_job_queue.get_job_status(
                trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)        
        return status_dict
