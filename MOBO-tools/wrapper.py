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

from ax.core.metric import Metric
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import matplotlib.pyplot as plt

import wandb

from ProjectUtils.ePICUtils.editxml import editGeom
from ProjectUtils.ePICUtils.wrapshell import piKsep, checkOverlap


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
    
    @glob_fun_2
    def getpiKsep(momentum, radiator):
        return RunSimulation(momentum,radiator)

    def getpiKsep_low(xdict):
        momentumVal = 14
        radiator = 0
        npart = 100        

        #edit geom, check overlaps
        for key in xdict:
            editGeom(key, xdict[key])
        n_overlap = checkOverlap()
        # TODO: -1 for failure in script, >0 for overlaps
        # do these need different responses?
        if n_overlap != 0:
            return -1

        #val = float(getpiKsep(momentumVal,radiator))
        val = piKsep(momentumVal,npart,radiator)
        return val
    def getpiKsep_high(xdict):
        momentumVal = 40
        radiator = 1
        npart = 100

        #edit geom, check overlaps
        for key in xdict:
            editGeom(key, xdict[key])
        n_overlap = checkOverlap()
        # TODO: -1 for failure in script, >0 for overlaps
        # do these need different responses?
        if n_overlap != 0:
            return -1

        # for some reason, using getpiKsep only calculates pi-K  
        # sep for the first set of params in a batch
        val = piKsep(momentumVal,npart,radiator)
        return val

    print(detconfig)
    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=i,
                           lower=float(detconfig[i]["lower"]), upper=float(detconfig[i]["upper"]), 
                           parameter_type=ParameterType.FLOAT)
            for i in detconfig],
        )

    # first test: nsigma pi-K separation at two momentum values
    names = ["piKsep_14GeV", "piKsep_60GeV"]
    functions = [getpiKsep_low, getpiKsep_high]
    metrics = []

    for name, function in zip(names, functions):
        metrics.append(
            GenericNoisyFunctionMetric(
                name=name, f=function, noise_sd=0.1, lower_is_better=False
            )
        )
    mo = MultiObjective(
        objectives=[Objective(m) for m in metrics],
        )
    objective_thresholds = [
        ObjectiveThreshold(metric=metrics[0], bound=2.5, relative=False),
        ObjectiveThreshold(metric=metrics[1], bound=2.5, relative=False)
        ]
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           objective_thresholds=objective_thresholds)

    # TODO: set real reference from current drich values
    ref_point = torch.tensor([2.5,2.5])
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
    
    if(not jsonFile):
        start_tot = time.time()
        experiment = build_experiment(search_space,optimization_config)
        start_gen = time.time()
        
        data = initialize_experiment(experiment,N_INIT)        
        end_gen = time.time()
        exp_df = exp_to_df(experiment)
        outcomes = torch.tensor(exp_df[names].values, **tkwargs)
        
        start_hv = time.time()
        partitioning = DominatedPartitioning(ref_point=ref_point, Y=outcomes)
        try:
            hv = partitioning.compute_hypervolume().item()
        except:
            hv = 0.
        end_hv = time.time()

        end_tot = time.time()
        time_tot.append(end_tot - start_tot)
        time_gen.append(end_gen - start_gen)
        time_hv.append(end_hv - start_hv)
        time_mcmc.append(-1.)
        time_trail.append(-1.)
        hv_list.append(hv)
        print(f"Initialized points, HV: {hv}")
        with open(os.path.join(outdir, "ax_state_init.json"), 'wb') as handle:
            list_dump = {"last_call": last_call,
                         "experiment": experiment,
                         #"HV_PARETO": hv_pareto,
                         "hv_list": hv_list,
                         "data": data,
                         "outcomes": outcomes,
                         "time_tot": time_tot,
                         "time_gen": time_gen,
                         "time_hv": time_hv,
                         "time_mcmc" : time_mcmc,
                         "time_trail" : time_trail
                         }
            pickle.dump(list_dump, handle, pickle.HIGHEST_PROTOCOL)
            print("saved initial generation file")
    if (jsonFile): 
        print("\n\n WARNING::YOU ARE LOADING AN EXISTING FILE: ", jsonFile, "\n\n")
        tmp_list = pickle.load(open(jsonFile, "rb" ))
        last_call = tmp_list["last_call"]
        experiment = tmp_list["experiment"]
        #hv_pareto = tmp_list["HV_PARETO"]
        hv_list = tmp_list["hv_list"]
        hv = hv_list[-1]
        data = tmp_list["data"]
        outcomes = tmp_list["outcomes"]
        time_tot = tmp_list["time_tot"]
        time_gen = tmp_list["time_gen"]
        time_hv = tmp_list["time_hv"]
        time_mcmc = tmp_list["time_mcmc"]
        time_trail = tmp_list["time_trail"]
        
    tol = config["hv_tolerance"]
    max_calls = config["max_calls"]
    #converged = (hv_pareto - hv)/hv_pareto 
    #converged_list.append(converged)
    check_imp = True
    roll = 30
    roll2 = min(len(hv_list)-1, 2*roll)
    if (len(hv_list) > roll):
        tmp_tol = 1. if hv_list[-roll]==0. else abs((hv_list[-1] - hv_list[-roll])/hv_list[-roll])

        # atleast 5% improvement w.r.t. last 5 calls and last call is better than first call
        check_imp = (tmp_tol > 0.0001) or (hv_list[-roll2] >= hv_list[-1]) #or (abs((hv_list[-1] - hv_list[1])/hv_list[1]) < 0.01)
        
    if (profiler):
        Profile_data = {"time_tot": time_tot,
                        "time_gen": time_gen,
                        "time_hv": time_hv,
                        "time_mcmc" : time_mcmc,
                        "time_trail" : time_trail,
                        "hv_list" : hv_list,
                        "converged_list" : converged_list
                        }
        pd.DataFrame(Profile_data).to_csv(os.path.join(outdir, "profile_data.csv"))
    if (doMonitor and jsonFile):
        logMetrics = {f"Trail Exec (q = {BATCH_SIZE}) [s]" : time_trail[-1],
                      "HV": hv,
                      #"Increase in HV w.r.t true pareto": converged,
                      "HV Calculation [s]": time_hv[-1],
                      "Total time [s]": time_tot[-1],
                      "iterations": last_call
                      }
        MLTracker.log(logMetrics)
    while(last_call <= max_calls and check_imp):
        start_tot = time.time()
        start_mcmc = time.time()
        model = Models.FULLYBAYESIANMOO(
            experiment=experiment,
            data=data,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            torch_device=tkwargs["device"],
            torch_dtype=tkwargs["dtype"],
            verbose=False,  # Set to True to print stats from MCMC
            disable_progbar=False,  # Set to False to print a progress bar from MCMC
        )
        end_mcmc = time.time()
        start_gen = time.time()
        generator_run = model.gen(BATCH_SIZE)
        end_gen = time.time()
        start_trail = time.time()
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()
        end_trail = time.time()
        
        data = Data.from_multiple_data([data, trial.fetch_data()])
        exp_df = exp_to_df(experiment)
        outcomes = torch.tensor(exp_df[names].values, **tkwargs)
        
        start_hv = time.time()
        partitioning = DominatedPartitioning(ref_point=ref_point, Y=outcomes)
        try:
            hv = partitioning.compute_hypervolume().item()
        except:
            hv = 0.
        
        end_hv = time.time()
        end_tot = time.time()
        
        last_call += 1
        #converged = (hv_pareto - hv)/hv_pareto
        hv_list.append(hv)
        if (len(hv_list) > roll):
            tmp_tol = 1. if(hv_list[-roll] == 0.) else abs((hv_list[-1] - hv_list[-roll])/hv_list[-roll])
            # atleast 5% improvement w.r.t. last #roll calls and last call is better than first call
            check_imp = (tmp_tol > 0.0001) or (hv_list[-roll2] >= hv_list[-1]) #or (abs((hv_list[-1] - hv_list[1])/hv_list[-1]) < 0.01)
        time_tot.append(end_tot - start_tot)
        time_mcmc.append(end_mcmc - start_mcmc)
        time_gen.append(end_gen - start_gen)
        time_trail.append(end_trail - start_trail)
        time_hv.append(end_hv - start_hv)
        #converged_list.append(converged)
        roll2+=1
        
        
        with open(os.path.join(outdir, optimInfo), "a") as f:
            f.write("Optimization call: " + str(last_call) + "\n")
            f.write("Optimization HV: " + str(hv) + "\n")
            #f.write(f"Optimization Pareto HV - HV / Pareto HV: {converged:.4f} \n")
            #f.write("Optimization converged: " + str(converged < tol) + "\n")
        
        if last_call % save_every_n == 0:
            with open(os.path.join(outdir, f'optim_iteration_{last_call}.json'), 'wb') as handle:
                list_dump = {"last_call": last_call,
                             "experiment": experiment,
                             #"HV_PARETO": hv_pareto,
                             "hv_list": hv_list,
                             "data": data,
                             "outcomes": outcomes,
                             "time_tot": time_tot,
                             "time_gen": time_gen,
                             "time_hv": time_hv,
                             "time_mcmc" : time_mcmc,
                             "time_trail" : time_trail
                             }
                pickle.dump(list_dump, handle)
                print(f"saved the file for {last_call} iteration")
            if (profiler):
                Profile_data = {"time_tot": time_tot,
                                "time_gen": time_gen,
                                "time_hv": time_hv,
                                "time_mcmc" : time_mcmc,
                                "time_trail" : time_trail,
                                "hv_list" : hv_list,
                                "converged_list" : converged_list
                                }
                pd.DataFrame(Profile_data).to_csv(os.path.join(outdir, "profile_data.csv"))
            if (doMonitor):
                logMetrics = {"MCMC Training [s]" : time_mcmc[-1],
                            f"Gen Acq func (q = {BATCH_SIZE}) [s]": time_gen[-1],
                            f"Trail Exec (q = {BATCH_SIZE}) [s]" : time_trail[-1],
                            "HV": hv,
                            #"Increase in HV w.r.t true pareto": converged,
                            "HV Calculation [s]": time_hv[-1],
                            "Total time [s]": time_tot[-1],
                            "iterations" : last_call
                            }
                MLTracker.log(logMetrics)
    MLTracker.finish()
        
        
  
