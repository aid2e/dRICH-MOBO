from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *

import os, pickle, torch, argparse, datetime
import time

import pandas as pd
from ax import *

import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.storage.json_store.save import save_experiment

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

import wandb

from ax.modelbridge.registry import Models
from ProjectUtils.runner_utilities import SlurmJobRunner
from ProjectUtils.metric_utilities import SlurmJobMetric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.modelbridge.torch import TorchModelBridge
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)

# for json storage of experiment
from ax.storage.registry_bundle import RegistryBundle

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
    parser.add_argument('-s', '--secret_file', 
                        help = "The file containing the secret key for weights and biases",
                        type = str, required = False,
                        default = "secrets.key")
    parser.add_argument('-p', '--profile',
                        help = "Profile the code",
                        type = bool, required = False, 
                        default = False)
    args = parser.parse_args()
    
    # READ SOME INFO 
    config = ReadJsonFile(args.config)
    detconfig = ReadJsonFile(args.detparameters)
    jsonFile = args.json_file
    profiler = args.profile
    outdir = config["OUTPUT_DIR"]    
                
    optimInfo = "optimInfo.txt" if not jsonFile else "optimInfo_continued.txt"
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    isGPU = torch.cuda.is_available()
    tkwargs = {
        "dtype": torch.double, 
        "device": torch.device("cuda" if isGPU else "cpu"),
    }
    with open(os.path.join(outdir, optimInfo), "w") as f:
        f.write("Optimization Info with name : " + config["name"] + "\n")
        f.write("Optimization has " + str(config["n_objectives"]) + " objectives\n")
        f.write("Optimization has " + str(config["n_design_params"]) + " design parameters\n")
        f.write("Optimization Info with description : " + config["description"] + "\n")
        f.write("Starting optimization at " + str(datetime.datetime.now()) + "\n")
        f.write(f"Optimization is running on {os.uname().nodename}\n")
        f.write("Optimization description : " + config["description"] + "\n")
        if(isGPU):
            f.write("Optimization is running on GPU : " + torch.cuda.get_device_name() + "\n")
    print ("Running on GPU? ", isGPU)
    
    print(detconfig["parameters"])

    def constraint_callable(text, parameters):
        def general_constraint(x):
            #x: pytorch tensor of design parameters
            values = {}
            for i, name in enumerate(parameters):            
                values[name] = x[...,i].item() 
            return eval(text, {}, values)
        return general_constraint

    def constraint_sobol(constraints,parameters):
        A = np.zeros( (len(constraints), len(parameters)) )
        b = np.zeros( (len(constraints), 1) )
        for i in range(len(constraints)):
            A[i][constraints[i][0]] = constraints[i][2]
            A[i][constraints[i][1]] = constraints[i][3]
            b[i] = constraints[i][4]
        return [A, b]

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
    print(constraints_ax)
    # create search space with linear constraints
    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=i,
                           lower=float(detconfig["parameters"][i]["lower"]), upper=float(detconfig["parameters"][i]["upper"]), 
                           parameter_type=ParameterType.FLOAT)
            for i in detconfig["parameters"]],
        parameter_constraints=constraints_ax        
    )
    print("made search space")
    # first test: nsigma pi-K separation at two momentum values
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
    objective_thresholds = [
        ObjectiveThreshold(metric=metrics[0], bound=2.5, relative=False),
        ObjectiveThreshold(metric=metrics[1], bound=2.5, relative=False),
        ObjectiveThreshold(metric=metrics[2], bound=0.7, relative=False)
        ]
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           objective_thresholds=objective_thresholds)

    # TODO: set real reference from current drich values?
    ref_point = torch.tensor([2.5, 2.5, 0.7])
    N_INIT = config["n_initial_points"]
    BATCH_SIZE = config["n_batch"]
    N_BATCH = config["n_calls"]
    num_samples = 64 if (not config.get("MOBO_params")) else config["MOBO_params"]["num_samples"]
    warmup_steps = 128 if (not config.get("MOBO_params")) else config["MOBO_params"]["warmup_steps"]
    
    hv_list = []
    time_gen = []
    time_mcmc = []
    time_hv = []
    time_tot = []
    time_trail = []
    converged_list = []
    hv = 0.0
    model = None
    last_call = 0
    
    start_tot = time.time()
    
    #experiment with custom slurm runner
    experiment = build_experiment_slurm(search_space,optimization_config, SlurmJobRunner())
    print("made experiment")
    gen_strategy = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                min_trials_observed=5,
                max_parallelism=5
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,            
                num_trials=-1,
                model_kwargs={  # args for BoTorchModel
                    "surrogate": Surrogate(botorch_model_class=SaasFullyBayesianSingleTaskGP,
                                           mll_options={"num_samples": 128,"warmup_steps": 256,  # Increasing this may result in better model fits
                                                        },
                                           ),
                    "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
                    "refit_on_update": True,
                    "refit_on_cv": True,
                    "warm_start_refit": True
                },
                max_parallelism=5
            ),
        ]
    )
    
    scheduler = Scheduler(experiment=experiment,
                          generation_strategy=gen_strategy,
                          options=SchedulerOptions())
    print("running BoTorch trials")
    scheduler.run_n_trials(max_trials=N_BATCH)
    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[names].values, **tkwargs)    
    exp_df.to_csv("dualmirror_df.csv")

    bundle = RegistryBundle(
        metric_clss={SlurmJobMetric: None}, runner_clss={SlurmJobRunner: None}
    )
    save_experiment(experiment=experiment,
                    filepath="dualmirror_experiment.json",
                    encoder_registry=bundle.encoder_registry)
    
