import sesame
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


J = []
for i in range(3):
    filename = "2dGB_IV_%d.gzip"%i
    sys, results = sesame.load_sim(filename)
    az = sesame.Analyzer(sys,results)
    J.append(az.full_current())
print(J)

##################

sys, results = sesame.load_sim("2dGB_V_0.gzip")
az = sesame.Analyzer(sys,results)

# Line endpoints of the grain boundary core
p1 = (20e-7, 1.5e-4)   #[cm]
p2 = (2.9e-4, 1.5e-4)  #[cm]
# get the coordinate indices of the grain boundary core
X, sites = az.line(sys, p1, p2)

# obtain solution data along the GB core
efn_GB = results['efn'][sites]
efp_GB = results['efp'][sites]
v_GB   = results['v'][sites]

#In this code we compute the integrated defect recombination along the grain boundary core::

# Get the first planar defect from the system
defect = sys.defects_list[0]
# Compute the defect recombination rate of this planar defect
R = az.defect_rr(defect)

# Compute the integrated recombination along the line defect
J = az.integrated_defect_recombination(defect)
