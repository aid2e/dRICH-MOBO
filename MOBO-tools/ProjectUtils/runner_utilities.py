from collections import defaultdict
from typing import Iterable, Set

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.core.trial import Trial
from .slurm_utilities import get_slurm_queue_client

class SlurmJobRunner(Runner):  # Deploys trials to external system.
    def run(self, trial: BaseTrial):
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")

        slurm_job_queue = get_slurm_queue_client()
        job_id = slurm_job_queue.schedule_job_with_parameters(
            parameters=trial.arm.parameters
        )
        return {"job_id": job_id}
    def poll_trial_status(self, trials: Iterable[BaseTrial]):
        status_dict = defaultdict(set)
        for trial in trials:
            slurm_job_queue = get_slurm_queue_client()
            status = slurm_job_queue.get_job_status(
                trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)        
        return status_dict
