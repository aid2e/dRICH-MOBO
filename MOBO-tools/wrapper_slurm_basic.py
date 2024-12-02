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
from ax.storage.registry_bundle import RegistryBundle

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Optimization, RICH_global")
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
                
    optimInfo = "optimInfo_RICH_global.txt" if not jsonFile else "optimInfo_continued.txt"
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
            constraint_list.append( ParameterConstraint(param_dict,constraints[c]["bound"]) )
        return constraint_list
    
    parameters = list(detconfig["parameters"].keys())
    # no constraints used currently
    # constraints_ax = constraint_ax(detconfig["constraints"],parameters)

    # create search space with[out] linear constraints
    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=i,
                           lower=float(detconfig["parameters"][i]["lower"]), upper=float(detconfig["parameters"][i]["upper"]), 
                           parameter_type=ParameterType.FLOAT)
            for i in detconfig["parameters"]] ) #, parameter_constraints=constraints_ax)

    # first test: mean of mchi2
    names = ["mean_mchi2"]
    metrics = []
    
    for name in names:
            metrics.append(SlurmJobMetric(name=name, lower_is_better=True))

    mo = MultiObjective(
        objectives=[Objective(m) for m in metrics],
        )
    objective_thresholds = [
        ObjectiveThreshold(metric=metrics[0], bound=3.0, relative=False),
        ]
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           objective_thresholds=objective_thresholds)

    # arm containing nominal design parameters to compare optimization points to
    status_quo_arm = Arm(
                    parameters={param : float(detconfig["parameters"][param]["default"]) for param in detconfig["parameters"]},
                    name="status_quo"
            )
    
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
    experiment = build_experiment_slurm(
        search_space=search_space,
        optimization_config=optimization_config,
        status_quo=status_quo_arm,
        runner=SlurmJobRunner()
    )
    
    gen_strategy = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=2,
                min_trials_observed=1,
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
                    # deprecated after ax-platform==0.4.0
                    # "refit_on_update": True,
                    "refit_on_cv": True,
                    "warm_start_refit": True
                },
                max_parallelism=5
            ),
        ]
    )
    
    # pre-calculated objective metric values for nominal design
    status_quo_metric_vals = [1.9]
    status_quo_data = Data(df=pd.DataFrame.from_records(
        [
            {
                "arm_name": "status_quo",
                "metric_name": names[i],
                "mean": metric_val,
                "sem": None,
                "trial_index": 0
            }
            for i, metric_val in enumerate(status_quo_metric_vals)
        ]
    ))

    # add data for status quo
    status_quo_trial = experiment.new_trial()
    status_quo_trial.add_arm(status_quo_arm)
    experiment.attach_data(status_quo_data)
    status_quo_trial.run().complete()
    
    scheduler = Scheduler(experiment=experiment,
                          generation_strategy=gen_strategy,
                          options=SchedulerOptions())
    print("running BoTorch trials")
    scheduler.run_n_trials(max_trials=N_BATCH)
    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[names].values, **tkwargs)    
    exp_df.to_csv("test_scheduler_df_rich_global.csv")
    
    # register custom metric and runner with json encoder
    bundle = RegistryBundle(
        metric_clss={SlurmJobMetric: None},
        runner_clss={SlurmJobRunner: None}
    )
    
    # export generation strategy model pkl file
    with open('gs_model_rich_global.pkl', 'wb') as file:
        pickle.dump(gen_strategy.model, file)

    # export experiment json file
    save_experiment(
        experiment=experiment,
        filepath='test_scheduler_experiment_rich_global.json',
        encoder_registry=bundle.encoder_registry
    )
