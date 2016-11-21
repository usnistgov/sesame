import sesame
import numpy as np
import sys as osys

#==============================================================================
#       System
#==============================================================================
# import builder
sys = sesame.Builder(T=100)

# define material(s) and geometry
Lx = 3e-6
Ly = 5e-6
Lz = 0
junction = 10e-9

nA = 1e15 * 1e6
nD = 1e17 * 1e6
CdTe = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
      'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 'RCenergy':0}
sys.add_material(((0,0,0), (Lx,Ly,Lz)), CdTe)
sys.add_donor(((0,0,0), (junction,Ly,Lz)), nD)
sys.add_acceptor(((junction,0,0), (Lx,Ly,Lz)), nA)
sys.contacts(1e48, 0, 0, 1e48)

# create mesh
x=np.concatenate((np.linspace(0,1.2e-6, 300, endpoint=False), 
                  np.linspace(1.2e-6, Lx, 100)))
y=np.concatenate((np.linspace(0, 2.25e-6, 100, endpoint=False), 
                  np.linspace(2.25e-6, 2.75e-6, 100, endpoint=False),
                  np.linspace(2.75e-6, Ly, 100)))
sys.mesh(x, y)

# define generation profile
# g0 = 1e27
# alpha = 5e7 # alpha = 5e5 cm^-1 for CdTe
# f = lambda x, y, z: g0 * alpha*Lx / (1 - np.exp(-alpha*Lx)) * np.exp(-alpha * x)
# sys.illumination(f)

# add local charges
S_GB = float(osys.argv[1]) * 1e-2             # trap recombination velocity
GBenergy = -0.25                                # energy of GB state
N_GB = 2e14 * 1e4                             # defect density. (1/m^2)

startGB = (20e-9, 2.5e-6, 0)
endGB = (2.8e-6, 2.5e-6, 0)
sys.add_local_charges([startGB, endGB], GBenergy, N_GB, S_GB)

# finalize the system
sys.finalize()


#==============================================================================
#       Calculation
#==============================================================================

# import matplotlib.pyplot as plt
# print(gtot2d/(2*np.pi*sigma**2) / sys.U)
# print(gtot2d * sys.xscale * sys.t0)
# g = np.reshape(sys.g, (sys.ny, sys.nx))
# plt.imshow(g)
# plt.colorbar()
# plt.show()


if __name__ == '__main__':
    # prepare the guess for the electrostatic potential
    v00 = np.log(abs(sys.rho[0])/sys.Nc[0])
    vL0 = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

    v = np.empty((sys.nx,), dtype=float)
    v[:sys.nx] = np.linspace(v00, vL0, sys.nx)
    v = np.tile(v, sys.ny)

    print("Electrostatics...")
    v = sesame.poisson_solver(sys, v, 1e-9, info=1, max_step=1000)
    # np.save('electrostatics.npy', [x, y, v])
    print("Electrostatics done")
    # _, _, v = np.load('electrostatics.npy')

    # _, _, efn, efp, v = np.load('bands.vapp_0.0.npy')
    efn = np.zeros((sys.nx*sys.ny,))
    efp = np.zeros((sys.nx*sys.ny,))
    for vapp in np.linspace(0, 40, 41):
        print(vapp)
        nx, ny = sys.nx, sys.ny
        for i in range(0, nx*(ny-1)+1, nx):
            v[i] = v00
            v[i+nx-1] = vL0 + vapp

        print("Solve DDP...")
        result = sesame.ddp_solver(sys, (efn, efp, v), 1e-9, max_step=30, info=1)
        print("DDP done")
        
        if result is None:
            print("no result for x0 = {0}".format(osys.argv[1]))
            exit(1)
        
        if result is not None:
            v = result['v']
            efn = result['efn']
            efp = result['efp']

            # np.save("dark-bands.S_{1}.vapp_{0}".format(vapp, osys.argv[1]), [x, y, efn, efp, v])
            efp = efp + 1
