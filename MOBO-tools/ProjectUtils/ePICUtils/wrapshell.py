import os
import xml.etree.ElementTree as ET
import sys
import subprocess

def piKsep(momentum, npart, radiator):
    shellcommand = [os.environ["EPIC_MOBO_UTILS"]+"/shell_wrapper.sh", str(momentum), str(1.5), str(3.5), str(npart), str(radiator)]

    commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)
    output = commandout.stdout.decode('utf-8')
    
    #print nsigma sep/objective on last line
    lines = output.split('\n')
    last_line = lines[-2] if lines else None    
    if last_line:
        line_split = last_line.split()
        if len(line_split) == 2:
            return float(line_split[1])
        else:
            return -1
    else:
        return -1
    return -1
