
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


p1 = (20e-7, 1.5e-4)
p2 = (2.9e-4, .5e-4)

X, sites = az.line(sys, p1, p2)

X = X * sys.scaling.length
v = results['v']*sys.scaling.energy
efn = results['efn']*sys.scaling.energy
efp = results['efp']*sys.scaling.energy
Eg = sys.Eg*sys.scaling.energy
plt.plot(X,-v[sites],'k')
plt.plot(X,-v[sites]-Eg[sites],'k')
plt.plot(X,efn[sites],'r')
plt.plot(X,-efp[sites],'b')
plt.show()




#az.map3D(result['v'])

#az.electron_current_map()


