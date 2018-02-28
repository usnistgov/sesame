import sesame
import numpy as np

results = np.load('1d_homojunction_jv_0.npz')

import matplotlib.pyplot as plt

vt = .0258
x0 = 1e4
plt.plot(results['x']*x0,-results['v']*vt,'k',linewidth=1.5)
plt.plot(results['x']*x0,-results['v']*vt-1.5,'k',linewidth=1.5)
plt.plot(results['x']*x0,results['efn']*vt,'b--',linewidth=1.5)
plt.plot(results['x']*x0,-results['efp']*vt,'r--',linewidth=1.5)
plt.xlabel('Position [/mu m]')
plt.ylabel('Energy [eV]')
plt.show()

