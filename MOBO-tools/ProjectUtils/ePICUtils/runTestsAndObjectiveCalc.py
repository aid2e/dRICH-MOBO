import numpy as np
import os, sys

#NEEDS TO:
# 1. run overlap check
# 2. generate p/eta scan
# 3. calculate pi-K sep
# 4. store all relevant results in drich-mobo-out_{jobid}.txt

nobj = 6
npart = 100
p_eta_scan = [
    [14, [1.3,2.0], 0],
    [14, [2.0,2.5], 0],
    [14, [2.5,3.5], 0],
    [40, [1.3,2.0], 1],
    [40, [2.0,2.5], 1],
    [40, [2.5,3.5], 1]
]
outname = str(os.environ["AIDE_HOME"])+"/log/results/"+ "drich-mobo-out_{}.txt".format(sys.argv[1])

def checkOverlap():
    shellcommand = [os.environ["EPIC_MOBO_UTILS"]+"/overlap_wrapper.sh"]

    commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
    output = commandout.stdout.decode('utf-8')

    lines = output.split('\n')
    last_line = lines[-2] if lines else None
    if last_line:
        line_split = last_line.split()
        if len(line_split) == 1:
            return int(line_split[0])
        else:
            return -1
    else:
        return -1
    return -1


noverlaps = checkOverlap()
if noverlaps != 0:
    # OVERLAP OR ERROR, return -1 for all objectives
    results = np.array( [-1 for i in range(nobj)])
    np.savetxt(outname,results)
    exit

results = []
for i in range(len(p_eta_scan)):
    momentum = p_eta_scan[i][0]
    eta_min = p_eta_scan[i][1][0]
    eta_max = p_eta_scan[i][1][1]
    radiator = p_eta_scan[i][2]
    shellcommand = [os.environ["EPIC_MOBO_UTILS"]+"/shell_wrapper_job.sh", str(momentum), str(eta_min), str(eta_max), str(npart), str(radiator)]
    
    commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
    output = commandout.stdout.decode('utf-8')

    piKsep = -1
    
    lines = output.split('\n')
    last_line = lines[-2] if lines else None
    if last_line:
        line_split = last_line.split()
        if len(line_split) == 2:
            piKsep = float(line_split[1])
        else:
            piKsep =  -1
    results.append(piKsep)

results = np.array(results)
np.savetxt(outname,results)
