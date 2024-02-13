from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *

import os, pickle, torch, argparse, datetime
import time

import pandas as pd
from ax import *

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

import wandb

from ProjectUtils.runner_utilities import SlurmJobRunner
from ProjectUtils.metric_utilities import SlurmJobMetric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
)
def RunSimulation(momentum,radiator):    
    # calculate objectives
    npart = 100
    # TODO: full p/eta scan
    result = piKsep(momentum,npart,radiator)
    return result

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
    save_every_n = config["save_every_n_call"]
    doMonitor = (True if config.get("WandB_params") else False) and profiler
    MLTracker = None
    if (doMonitor):
        if (not os.getenv("WANDB_API_KEY") and not os.path.exists(args.secret_file)):
            print ("Please set WANDB_API_KEY in your environment variables or include a file named secrets.key in the same directory as this script.")
            sys.exit()
        else:
            os.environ["WANDB_API_KEY"] = ReadJsonFile(args.secret_file)["WANDB_API_KEY"] if not os.getenv("WANDB_API_KEY") else os.environ["WANDB_API_KEY"]
            wandb.login(anonymous='never', key = os.environ['WANDB_API_KEY'], relogin=True)
            track_config = {"n_design_params": config["n_design_params"], "n_objectives" : config["n_objectives"]}
            MLTracker = wandb.init(config = track_config, **config["WandB_params"])
            
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
    
    print(detconfig)
    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=i,
                           lower=float(detconfig[i]["lower"]), upper=float(detconfig[i]["upper"]), 
                           parameter_type=ParameterType.FLOAT)
            for i in detconfig],
        )

    # first test: nsigma pi-K separation at two momentum values
    names = ["piKsep_plow_etalow",
             "piKsep_plow_etamid",
             "piKsep_plow_etahigh",
             "piKsep_phigh_etalow",
             "piKsep_phigh_etamid",
             "piKsep_phigh_etahigh"
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
        ObjectiveThreshold(metric=metrics[i], bound=2.5, relative=False) for i in range(len(names))
        ]
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           objective_thresholds=objective_thresholds)

    # TODO: set real reference from current drich values
    ref_point = torch.tensor([2.5 for i in range(len(names))])
    N_INIT = config["n_initial_points"]
    BATCH_SIZE = config["n_batch"]
    N_BATCH = config["n_calls"]
    num_samples = 64 if (not config.get("MOBO_params")) else config["MOBO_params"]["num_samples"]
    warmup_steps = 128 if (not config.get("MOBO_params")) else config["MOBO_params"]["warmup_steps"]
    if (doMonitor):
        MLTracker.config["BATCH_SIZE"] = BATCH_SIZE
        MLTracker.config["N_BATCH"] = N_BATCH
        MLTracker.config["num_samples"] = num_samples
        MLTracker.config["warmup_steps"] = warmup_steps
        MLTracker.define_metric("iterations")
        logMetrics = ["MCMC Training [s]", 
                      f"Gen Acq func (q = {BATCH_SIZE}) [s]",
                      f"Trail Exec (q = {BATCH_SIZE}) [s]",
                      "HV",
                      "Increase in HV w.r.t true pareto",
                      "HV Calculation [s]",
                      "Total time [s]"]
        for l in logMetrics:
            MLTracker.define_metric(l, step_metric = "iterations")
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
    # Generate initial number of SOBOL points
    initial_generation = GenerationStep(model = Models.SOBOL, num_trials = N_INIT, min_trials_observed = N_INIT, max_parallelism=5)
    # The Surrogate here is SAASBO with qNEHVI acq. 
    # TO DO: Need to play with the hyper parameters here
    model = BoTorchModel(
        surrogate = Surrogate(
        botorch_model_class=SaasFullyBayesianSingleTaskGP,
        mll_options={
            "num_samples": 256,  # Increasing this may result in better model fits
            "warmup_steps": 512,  # Increasing this may result in better model fits
                    },
                ),
        botorch_acqf_class = qNoisyExpectedImprovement,
        acquisition_options = {},
        refit_on_update = True, 
        refit_on_cv = False, 
        warm_start_refit = True
    )
    subsequent_generation = GenerationStep(model = model, 
                                           num_trials = BATCH_SIZE,
                                           min_trials_observed = BATCH_SIZE
                                           )
    gen_strategy = GenerationStrategy(steps = [initial_generation, subsequent_generation])
    scheduler = Scheduler(experiment=experiment,
                          generation_strategy=gen_strategy,
                          options=SchedulerOptions())
    scheduler.run_n_trials(max_trials=1)

    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[names].values, **tkwargs)
    print("successful outcomes: ", outcomes)
    exp_df.to_csv("test_scheduler_df.csv")
    
