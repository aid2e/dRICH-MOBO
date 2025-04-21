from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *

import os, pickle, torch, argparse, datetime
import time
import logging

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
# from ProjectUtils.panda_idds_utilities import build_experiment_pandaidds, PanDAIDDSJobRunner, PanDAIDDSJobMetric
from ProjectUtils.panda_idds_utilities_parallel import build_experiment_pandaidds, PanDAIDDSJobRunner, PanDAIDDSJobMetric
from ProjectUtils.panda_idds_utilities_multi_steps import (build_experiment_pandaidds as build_experiment_pandaidds_mul,
        PanDAIDDSJobRunner as PanDAIDDSJobRunner_mul, PanDAIDDSJobMetric as PanDAIDDSJobMetric_mul)

from ProjectUtils.local_utilities import build_experiment_local, LocalJobRunner, LocalJobMetric

from ax.modelbridge.registry import Models
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
    qNoisyExpectedHypervolumeImprovement,
)

# for sql storage of experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

from ax.storage.registry_bundle import RegistryBundle
from ax.storage.sqa_store.db import (
    create_all_tables,
    get_engine,
    init_engine_and_session_factory,
)
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings


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
    parser.add_argument('-b', '--backend',
                        help = "The backend to run the jobs (slurm, panda, local)",
                        type = str, required = False,
                        default = 'local')
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
            for i in detconfig["parameters"]]
        #,
        # FOR NOW, reparamterized so no constraints on dRICH geometry
        #parameter_constraints=constraints_ax        
    )

    names = ["piKsep_etalow",            
             "piKsep_etahigh",
             "acceptance"
             ]  
    metrics = []

    if args.backend == 'slurm':
        for name in names:
            metrics.append(SlurmJobMetric(name=name, lower_is_better=False))
    elif args.backend == 'panda':
        for name in names:
            metrics.append(PanDAIDDSJobMetric(name=name, lower_is_better=False))
    elif args.backend == 'panda_multi_steps':
        for name in names:
            metrics.append(PanDAIDDSJobMetric_mul(name=name, lower_is_better=False))
    else:
        for name in names:
            metrics.append(LocalJobMetric(name=name, lower_is_better=False))

    mo = MultiObjective(
        objectives=[Objective(m) for m in metrics],
        )
    # 10% below what would be acceptable design
    objective_thresholds = [
        ObjectiveThreshold(metric=metrics[0], bound=2.7, relative=False),
        ObjectiveThreshold(metric=metrics[1], bound=2.7, relative=False),
        ObjectiveThreshold(metric=metrics[2], bound=0.6, relative=False)
        ]
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           objective_thresholds=objective_thresholds)

    #TODO: implement check of dominated HV convergence instead of
    #fixed N points
    BATCH_SIZE_SOBOL = config["n_batch_sobol"]
    BATCH_SIZE_MOBO = config["n_batch_mobo"]
    BATCH_SIZE = max(BATCH_SIZE_SOBOL, BATCH_SIZE_MOBO)
    N_SOBOL = config["n_sobol"]
    N_MOBO = config["n_mobo"]
    N_TOTAL = N_SOBOL + N_MOBO
    if (N_MOBO == -1) or (N_SOBOL==-1):
        N_TOTAL+=1
    print("running ", N_TOTAL, " trials")
    outname = config["OUTPUT_NAME"]
    #N_BATCH = config["n_calls"]
    num_samples = 64 if (not config.get("MOBO_params")) else config["MOBO_params"]["num_samples"]
    warmup_steps = 128 if (not config.get("MOBO_params")) else config["MOBO_params"]["warmup_steps"]
    
    print(f"backend: {args.backend}")
    print(f"BATCH_SIZE_SOBOL: {BATCH_SIZE_SOBOL}")
    print(f"BATCH_SIZE_MOBO: {BATCH_SIZE_MOBO}")
    print(f"N_SOBOL: {N_SOBOL}")
    print(f"N_MOBO: {N_MOBO}")

    if args.backend == 'slurm':
        # experiment with custom slurm runner
        experiment = build_experiment_slurm(search_space, optimization_config, SlurmJobRunner())
        register_runner(SlurmJobRunner)
        register_metric(SlurmJobMetric)

        bundle = RegistryBundle(
            metric_clss={SlurmJobMetric: None}, runner_clss={SlurmJobRunner: None}
        )
    elif args.backend == 'panda':
        experiment = build_experiment_pandaidds(search_space, optimization_config, PanDAIDDSJobRunner())
        register_runner(PanDAIDDSJobRunner)
        register_metric(PanDAIDDSJobMetric)

        bundle = RegistryBundle(
            metric_clss={PanDAIDDSJobMetric: None}, runner_clss={PanDAIDDSJobRunner: None}
        )
    elif args.backend == 'panda_multi_steps':
        experiment = build_experiment_pandaidds_mul(search_space, optimization_config, PanDAIDDSJobRunner_mul())
        register_runner(PanDAIDDSJobRunner_mul)
        register_metric(PanDAIDDSJobMetric_mul)

        bundle = RegistryBundle(
            metric_clss={PanDAIDDSJobMetric_mul: None}, runner_clss={PanDAIDDSJobRunner_mul: None}
        )
    else:
        experiment = build_experiment_local(search_space, optimization_config, LocalJobRunner())
        register_runner(LocalJobRunner)
        register_metric(LocalJobMetric)

        bundle = RegistryBundle(
            metric_clss={LocalJobMetric: None}, runner_clss={LocalJobRunner: None}
        )

    # set up sql storage
    db_settings = DBSettings(
        url="sqlite:///{}.db".format(outname),
        encoder=bundle.encoder,
        decoder=bundle.decoder
    )
    init_engine_and_session_factory(url=db_settings.url)
    engine = get_engine()
    create_all_tables(engine)

    gen_strategy = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=N_SOBOL,
                min_trials_observed=N_SOBOL,
                max_parallelism=BATCH_SIZE_SOBOL,
                model_kwargs={"seed": 999}
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,            
                num_trials=N_MOBO,
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
                max_parallelism=BATCH_SIZE_MOBO
            ),
        ]
    )
    
    scheduler = Scheduler(experiment=experiment,
                          generation_strategy=gen_strategy,
                          options=SchedulerOptions(max_pending_trials=BATCH_SIZE, logging_level=logging.DEBUG),
                          db_settings=db_settings)

    print("before run_n_trials")
    scheduler.run_n_trials(max_trials=N_TOTAL)
    print("after run_n_trials")

    model_obj = Models.BOTORCH_MODULAR(experiment = experiment, data = experiment.fetch_data())

    # TODO: check for HV convergence
    hv = observed_hypervolume(modelbridge=model_obj)
    print(f"hv: {hv}")
    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[names].values, **tkwargs)    
    exp_df.to_csv(outname+".csv")

