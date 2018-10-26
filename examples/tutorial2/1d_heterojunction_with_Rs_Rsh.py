import sesame
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# define contact and shunt resistance-area values (note units)
Rs = 3  # Ohm-cm^2
Rsh = 500  # Ohm-cm^2

# load previous J-V data
result = np.load('IV_values.npy')
voltages = result.tolist()['v']
j = result.tolist()['j']

# steps:  make cubic spline of J-V.  We need a nearly continuous representation of diode J-V
voltages_finegrid = np.linspace(0,1,1000)
# interpolation function
j_interp = interp1d(voltages, j, kind='cubic')
# make interpolated current
j_finegrid = j_interp(voltages_finegrid)

# get numerically computed short-circuit current
jsc = j_finegrid[0]

# jdiode returns numerically computed (dark) current for given voltage
def jdiode(v):
    vind = np.argmin(np.abs(v - voltages_finegrid))
    return(np.abs(j_finegrid[vind] - jsc))


# declare array to store current of equivalent circuit model
j_withcontacts = np.zeros(len(voltages_finegrid))

# cycle over applied voltage values
for c, vapp in enumerate(voltages_finegrid):

    # for fixed voltage, define equivalent circuit current-voltage equation
    equivCircuit = lambda j: np.abs(j + jdiode(vapp + j*Rs) + ((vapp + j*Rs)/Rsh) - jsc)
    # find zero of the (nonlinear) equivalent circuit relaction
    thisj = minimize(equivCircuit, j_finegrid[c]/2, tol=1e-6,  method='SLSQP')
    # store answer
    j_withcontacts[c] = thisj.x


# plot I-V curve
try:
    import matplotlib.pyplot as plt
    plt.plot(voltages_finegrid, 1e3*j_finegrid, '-', linewidth=2.0)
    plt.plot(voltages_finegrid, 1e3*j_withcontacts, '--', linewidth=2.0)
    plt.ylabel(r'J [$\mathregular{mA/cm^2}$]')
    plt.xlabel('V [V]')
    plt.grid()
    plt.ylim((-10, 20))
    plt.legend((r'$R_s$=0 $\mathregular{\Omega\cdot cm^2}$, $R_{sh}$=$\infty\ \mathregular{\Omega\cdot cm^2}$',\
                r'$R_s$=3 $\mathregular{\Omega\cdot cm^2}$, $R_{sh}$=$500\ \mathregular{\Omega\cdot cm^2}$'),loc='lower left')
    plt.show()

except ImportError:
    print("Matplotlib not installed, can't make plot")


