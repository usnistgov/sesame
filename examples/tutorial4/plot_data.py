import sesame
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


sys, results = sesame.load_sim("2dGB_V_0.gzip")
az = sesame.Analyzer(sys,results)

az.map3D(results['v']) # units of kT/q

az.electron_current_map()

p1 = (2e-4, 0)
p2 = (2e-4, 3e-4)

X, sites = az.line(sys, p1, p2)
X = sys.scaling.length * X
# For the entire system
n2d = az.electron_density()
n2d = n2d * sys.scaling.density
n2d = np.reshape(n2d, (sys.ny, sys.nx))


# On the previously defined line
n1d = az.electron_density((p1, p2))
n1d = n1d * sys.scaling.density


plt.figure(1)
plt.plot(X, np.log(n1d))
plt.xlabel('Position [cm]')
plt.ylabel('ln[n]')
plt.title('ln(n) across GB')

plt.figure(2)
plt.contourf(sys.xpts, sys.ypts, np.log(n2d))
plt.xlabel('Position [cm]')
plt.ylabel('Position [cm]')
plt.colorbar()
plt.title('ln(n)')

plt.show() # show the figures on the screen

az.band_diagram((p1,p2))

