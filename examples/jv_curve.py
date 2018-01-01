import sesame
import numpy as np

def system(amp=1):
    # Dimensions of the system
    Lx = 3e-6 # [m]
    Ly = 5e-6 # [m]
    # extent of the junction from the left contact [m]
    junction = 10e-9 

    # Mesh
    x = np.concatenate((np.linspace(0,1.2e-6, 100, endpoint=False), 
                        np.linspace(1.2e-6, 2.9e-6, 50, endpoint=False),
                        np.linspace(2.9e-6, Lx, 10)))
    y = np.concatenate((np.linspace(0, 2.25e-6, 50, endpoint=False), 
                        np.linspace(2.25e-6, 2.75e-6, 50, endpoint=False),
                        np.linspace(2.75e-6, Ly, 50)))

    sys = sesame.Builder(x, y, input_length='m')

    # Add the donors
    nD = 1e17 * 1e6 # [m^-3]
    sys.add_donor(nD, lambda pos: pos[0] < junction)

    # Add the acceptors
    nA = 1e15 * 1e6 # [m^-3]
    sys.add_acceptor(nA, lambda pos: pos[0] >= junction)

    # Use perfectly selective contacts
    Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
    sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

    # Region 1
    reg1 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9}
    sys.add_material(reg1, lambda pos: (pos[1] <= 2.4e-6) | (pos[1] >= 2.6e-6))

    # Region 2
    reg2 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':20*1e-4, 'mu_h':20*1e-4, 'tau_e':10e-9, 'tau_h':10e-9}
    sys.add_material(reg2, lambda pos: (pos[1] > 2.4e-6) & (pos[1] < 2.6e-6))

    # gap state characteristics
    s = 1e-15 * 1e-4               # trap capture cross section [m^2]
    E = -0.25                      # energy of gap state (eV) from midgap
    N = 2e13 * 1e4           # defect density [1/m^2]

    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]

    sys.add_line_defects([p1, p2], N, s, E=E)

    # Define a function for the generation rate
    phi = amp * 1e21          # photon flux [1/(m^2 s)]
    alpha = 2.3e6       # absorption coefficient [1/m]
    f = lambda x, y: phi * alpha * np.exp(-alpha * x)
    sys.generation(f)

    return sys



if __name__ == '__main__':

    # Compute the equilibrium potential (just Poisson equation)
    sys = system()
    solution = sesame.solve_equilibrium(sys)

    # Loop at zero bias with increasing generation rate
    for amp in [0.001, 0.01, 0.05, 0.1, 0.5]:
        print("amplitude: ", amp)
        asys = system(amp)
        solution = sesame.solve(asys, solution)

    # Loop over the applied potentials for the desired system
    voltages = np.linspace(0, 1, 40)
    sesame.IVcurve(sys, voltages, solution, '2dpnIV.vapp')
