from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *

import os, pickle, torch, argparse, datetime
import time

import pandas as pd
from ax import *
from ax.modelbridge.registry import Models
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp

import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

# Scheduler imports
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import Scheduler, SchedulerOptions

from ax.core.metric import Metric
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import matplotlib.pyplot as plt


from ax.modelbridge.registry import Models
from ProjectUtils.runner_utilities import SlurmJobRunner
from ProjectUtils.metric_utilities import SlurmJobMetric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.modelbridge.torch import TorchModelBridge
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

# for sql storage of experiment
from ax.storage.metric_registry import register_metrics
from ax.storage.runner_registry import register_runner

#early stopping
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Optimization, dRICH")
    parser.add_argument('-c', '--config', 
                        help='Optimization configuration file', 
                        type = str, required = True)
    parser.add_argument('-d', '--detparameters', 
                        help='Detector parameter configuration file', 
                        type = str, required = True)
    parser.add_argument('-j', '--json_file', 
                        help = "The json file to load and continue optimization", 
                        type = str, required=False)
    args = parser.parse_args()
    
    config = ReadJsonFile(args.config) # optimization parameters
    detconfig = ReadJsonFile(args.detparameters) # geometry parameters

    outdir = config["OUTPUT_DIR"]    
    # create specified output directory if it doesn't exist
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    isGPU = torch.cuda.is_available()
    tkwargs = {
        "dtype": torch.double, 
        "device": torch.device("cuda" if isGPU else "cpu"),
    }

    # creates linear constraint to pass to ax SearchSpace
    # based on list of parameters with weights given in --detparameters.
    # constraints pass if output < 0
    def constraint_ax(constraints,parameters):
        # constraint_dict: Dict[str,float], bound: float
        constraint_list = []
        for c in constraints:
            param_dict = {}
            param_list = constraints[c]["parameters"]
            for param in parameters:
                if param in param_list:
                    param_dict[param] = constraints[c]["weights"][param_list.index(param)]
                else:
                    param_dict[param] = 0
            print("param dict: ", param_dict, " param_list: ", param_list)
            constraint_list.append( ParameterConstraint(param_dict,constraints[c]["bound"]) )
        return constraint_list    
    parameters = list(detconfig["parameters"].keys())
    constraints_ax = constraint_ax(detconfig["constraints"],parameters)

    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=i,
                           lower=float(detconfig["parameters"][i]["lower"]), upper=float(detconfig["parameters"][i]["upper"]), 
                           parameter_type=ParameterType.FLOAT)
            for i in detconfig["parameters"]]
        #,
        # currently reparamterized so no constraints on dRICH geometry,
        # but could be applied to search space here.
        #parameter_constraints=constraints_ax
    )

    # TODO: find some better way to pass these between here,
    # runner, and objective calculation script.
    names = ["piKsep_etalow",
             "piKsep_etahigh",
             "acceptance"
             ]
    metrics = []
    
    for name in names:
        metrics.append(
            SlurmJobMetric(
                name=name, lower_is_better=False
            )
        )
    mo = MultiObjective(
        objectives=[Objective(m) for m in metrics],
    )
    # 10% below what would be acceptable design (here nominal dRICH performance)
    #ETA BINS:
    objective_thresholds = [
        ObjectiveThreshold(metric=metrics[0], bound=3.62, relative=False),
        ObjectiveThreshold(metric=metrics[1], bound=4.04, relative=False),
        ObjectiveThreshold(metric=metrics[2], bound=0.80, relative=False)
    ]

    # TODO: figure out how to make OutcomeConstraint work correctly with Scheduler
    #outcome_constraint = OutcomeConstraint(metrics[2],ComparisonOp.GEQ,0.75)
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           #outcome_constraints=[outcome_constraint],
                                                           objective_thresholds=objective_thresholds)

    #TODO: implement check of dominated HV convergence instead of
    #fixed N points
    BATCH_SIZE_SOBOL = config["n_batch_sobol"]
    BATCH_SIZE_MOBO = config["n_batch_mobo"]
    N_SOBOL = config["n_sobol"]
    N_MOBO = config["n_mobo"]
    N_TOTAL = N_SOBOL + N_MOBO
    if (N_MOBO == -1) or (N_SOBOL==-1):
        N_TOTAL+=1
    print("Scheduling ", N_TOTAL, " trials")

    outname = config["OUTPUT_NAME"]
    num_samples = 64 if (not config.get("MOBO_params")) else config["MOBO_params"]["num_samples"]
    warmup_steps = 128 if (not config.get("MOBO_params")) else config["MOBO_params"]["warmup_steps"]
    
    #experiment with custom slurm runner
    experiment = build_experiment_slurm(search_space, optimization_config, SlurmJobRunner(metrics=names))
    
    gen_strategy = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=N_SOBOL,
                max_parallelism=BATCH_SIZE_SOBOL,
                model_kwargs={"seed": 999}
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,            
                num_trials=-1,
                model_kwargs={  # args for BoTorchModel
                    "surrogate": Surrogate(
                        botorch_model_class=SingleTaskGP
                    ),
                    "botorch_acqf_class": qLogNoisyExpectedHypervolumeImprovement,
                    "refit_on_cv": True,
                    "warm_start_refit": True
                },
                max_parallelism=BATCH_SIZE_MOBO
            ),
        ]
    )
    
    # setting up early stopping strategy
    stopping_strategy = ImprovementGlobalStoppingStrategy(
        min_trials= N_SOBOL + 5*BATCH_SIZE_MOBO, window_size=3*BATCH_SIZE_MOBO, improvement_bar=0.01
    )
    
    scheduler = Scheduler(experiment=experiment,
                          generation_strategy=gen_strategy,
                          options=SchedulerOptions(init_seconds_between_polls=10,
                                                   seconds_between_polls_backoff_factor=1,
                                                   min_failed_trials_for_failure_rate_check=5,                                                   
                                                   #global_stopping_strategy=stopping_strategy
                                                   )
                          )

    scheduler.run_n_trials(max_trials=N_TOTAL)
    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[names].values, **tkwargs)
    exp_df.to_csv(outdir+"/"+outname+".csv")

    # save model object for further analysis
    with open(outdir+"/"+outname+'_gs_model.pkl', 'wb') as file:
        pickle.dump(gen_strategy.model, file) 
   
