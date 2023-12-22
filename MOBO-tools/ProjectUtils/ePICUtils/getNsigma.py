import os
import numpy as np
import sys
import math

# nphotons, Ch_mean, Ch_sigma, chi square
pi_plus_cher = np.loadtxt(sys.argv[1])
K_plus_cher = np.loadtxt(sys.argv[2])

mean_nphot = (pi_plus_cher[0] + K_plus_cher[0])/2
mean_sigma = (pi_plus_cher[2] + K_plus_cher[2])/2
nsigma = (abs(pi_plus_cher[1] - K_plus_cher[1])*math.sqrt(mean_nphot))/mean_sigma
#np.savetxt("nsigmasep.txt",nsigma)
print("NsigmaSep ", nsigma)
