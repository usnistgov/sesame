import numpy as np
import scipy.constants as cts
from collections import namedtuple
from sesame.utils2 import get_indices

class Builder():
    def __init__(self, T=300):
        # temperature
        self.T = T

        # scalings for...
        # densities
        self.N = 1e19 * 1e6 # [m^-3]
        # energies
        self.vt = cts.k * T / cts.e
        # mobilities [m^2 / (V.s)]
        self.mu = 1e-4
        # time [s]
        self.t0 = cts.epsilon_0 / (self.mu * cts.e* self.N)
        # space [m]
        self.xscale = np.sqrt(cts.epsilon_0 * self.vt / (cts.e * self.N))
        # generation rate [m^-3]
        self.U = (self.N * self.mu * self.vt) / self.xscale**2 
        # recombination velocities
        self.Sc = self.xscale / self.t0

        # list of the regions in the system
        self.region = namedtuple('region', ['xa', 'ya', 'za', 'xb', 'yb', 'zb', 'material'])
        self.regions = []
        self.dopant = namedtuple('dopant', ['xa', 'ya', 'za', 'xb', 'yb', 'zb', 'density'])
        self.dopants = []
        self.charge = namedtuple('charge', ['xa', 'ya', 'za', 'xb', 'yb', 'zb',
                                            'energy', 'density', 'S'])
        self.charges = []

        # generation of carriers
        self.illumination = None
        # length of mesh in x, y, z directions
        self.nx, self.ny, self.nz = 1, 1, 1
        self.dimension = 1



    def add_material(self, location, mat):
        """
        Update self.regions with dimensionless material parameters

        Arguments
        ---------
        location: x,y,z-coordinates of the material region [m]. Tuple containing
        x,y,z, use zeros for unused dimensions.
        mat: dictionary with the material parameters
        """

        # make material parameters dimensionless
        scale = {'Nc':self.N, 'Nv':self.N, 'Eg':self.vt, 'epsilon':1,
                 'mu_e':self.mu, 'mu_h':self.mu,
                 'tau_e':self.t0, 'tau_h':self.t0,
                 'RCenergy':self.vt}
        mat = {k: mat[k] / scale[k] for k in mat.keys() & scale.keys()}

        # create a named_tuple for the region and update the list of regions
        xa, ya, za = location[0]
        xb, yb, zb = location[1]
        r = self.region(xa/self.xscale, ya/self.xscale, za/self.xscale,
                        xb/self.xscale, yb/self.xscale, zb/self.xscale, mat)
        self.regions.append(r)

    def add_local_charges(self, location, local_E, local_N, local_S):
        """
        Add charges (for a grain boundary for instance) to the total charge rho

        Arguments
        ---------
        location: coordinates of the two points defining a line in a 2D problem
        or a rectangle in a 3D problem [m]. List of two tuples containing x,y,z.
        Use zeros for unused dimensions.
        local_E: energy level of the states defined with respect to Eg/2 [eV]
        local_N: defect density [m^-2]
        local_S: surface recombination velocity [m.s^-1]
        """
        
        xa, ya, za = location[0]
        xb, yb, zb = location[1]
        
        d = self.charge(xa/self.xscale, ya/self.xscale, za/self.xscale,
                        xb/self.xscale, yb/self.xscale, zb/self.xscale,
                        local_E/self.vt, local_N/(self.N*self.xscale), 
                        local_S/self.Sc)
        self.charges.append(d)

    def doping_profile(self, location, density):
        """
        Add dopant charges to the system

        Arguments
        ---------
        location: x,y,z-coordinates of the material region [m]. Tuple containing
        x,y,z, use zeros for unused dimensions.
        density: doping density [m^-3]
        """

        xa, ya, za = location[0]
        xb, yb, zb = location[1]

        d = self.dopant(xa/self.xscale, ya/self.xscale, za/self.xscale,
                        xb/self.xscale, yb/self.xscale, zb/self.xscale, 
                        density/self.N)
        self.dopants.append(d)

    def add_donor(self, location, density):
        self.doping_profile(location, density)

    def add_acceptor(self, location, density):
        self.doping_profile(location, -density)

    def illumination(self, f):
        """
        Illumination profile along x, assumed invariant along y

        Arguments
        ---------
        f: function for the generation rate [m^-3]
        """

        self.illumination = lambda x: f(x) / self.U

    def contacts(self, Scn_left, Scp_left, Scn_right, Scp_right):
        """
        Create the lists of recombination velocities for the contacts

        Arguments
        ---------
        Scx: recombination velocities [m.s^-1]
        """

        self.Scn = [Scn_left/self.Sc, Scn_right/self.Sc]
        self.Scp = [Scp_left/self.Sc, Scp_right/self.Sc]

    def mesh(self, xpts, ypts=None, zpts=None):
        """
        User provided mesh of the system

        Arguments
        ---------
        xpts, ypts, zpts: numpy arrays [m]
        """

        self.xpts = xpts / self.xscale
        self.dx = self.xpts[1:] - self.xpts[:-1]
        self.nx = xpts.shape[0]

        self.ypts = ypts
        if ypts is not None:
            self.ypts = ypts / self.xscale
            self.dy = self.ypts[1:] - self.ypts[:-1]
            self.ny = ypts.shape[0]
            self.dimension = 2

        self.zpts = zpts
        if zpts is not None:
            self.zpts = zpts / self.xscale
            self.dz = self.zpts[1:] - self.zpts[:-1]
            self.nz = zpts.shape[0]
            self.dimension = 3

    def finalize(self):
        """
        Generate the local charge for all points in the system, create
        arrays of materials parameters
        """

        # mesh parameters
        nx, ny, nz = self.nx, self.ny, self.nz

        def get_sites(xa, ya, za, xb, yb, zb):
            s = [i + j*nx + k*nx*ny for k in range(za, zb+1) 
                                    for j in range(ya, yb+1) 
                                    for i in range(xa, xb+1)]
            return s


        # materials properties
        self.Nc = np.zeros((nx*ny*nz,), dtype=float)
        self.Nv = np.zeros((nx*ny*nz,), dtype=float)
        self.Eg = np.zeros((nx*ny*nz,), dtype=float)
        self.epsilon = np.zeros((nx*ny*nz,), dtype=float)
        self.mu_e = np.zeros((nx*ny*nz,), dtype=float)
        self.mu_h = np.zeros((nx*ny*nz,), dtype=float)
        self.tau_e = np.zeros((nx*ny*nz,), dtype=float)
        self.tau_h = np.zeros((nx*ny*nz,), dtype=float)
        self.n1 = np.zeros((nx*ny*nz,), dtype=float)
        self.p1 = np.zeros((nx*ny*nz,), dtype=float)
        for r in self.regions:
            xa, ya, za = get_indices(self, (r.xa, r.ya, r.za))
            xb, yb, zb = get_indices(self, (r.xb, r.yb, r.zb))
            s = get_sites(xa, ya, za, xb, yb, zb)

            self.Nc[s] = r.material['Nc']
            self.Nv[s] = r.material['Nv']
            self.Eg[s] = r.material['Eg']
            self.epsilon[s] = r.material['epsilon']
            self.mu_e[s] = r.material['mu_e']
            self.mu_h[s] = r.material['mu_h']
            self.tau_e[s] = r.material['tau_e']
            self.tau_h[s] = r.material['tau_h']
            self.n1[s] = self.Nc[s] * np.exp(-self.Eg[s]/2 + r.material['RCenergy'])
            self.p1[s] = self.Nv[s] * np.exp(-self.Eg[s]/2 - r.material['RCenergy'])
        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg/2)

        # set the electrostatic charge from the doping profile
        self.rho = np.zeros((nx*ny*nz,), dtype=float)
        for d in self.dopants:
            xa, ya, za = get_indices(self, (d.xa, d.ya, d.za))
            xb, yb, zb = get_indices(self, (d.xb, d.yb, d.zb))
            s = get_sites(xa, ya, za, xb, yb, zb)
            self.rho[s] = d.density

        # additional extra charges
        if len(self.charges) != 0:
            self.Nextra = np.zeros((nx*ny*nz,), dtype=float)
            self.Sextra = np.zeros((nx*ny*nz,), dtype=float)
            self.nextra = np.zeros((nx*ny*nz,), dtype=float)
            self.pextra = np.zeros((nx*ny*nz,), dtype=float)
            self.extra_charge_sites = []
            for c in self.charges:
                xa, ya, za = get_indices(self, (c.xa, c.ya, c.za))
                xb, yb, zb = get_indices(self, (c.xb, c.yb, c.zb))
                if xa == xb: # parallel to the junction
                    dl = xpts[xa+1] - xpts[xa]
                elif ya == yb: # orthogonal to the junction
                    dl = ypts[ya+1] - ypts[ya]

                else:
                    s = get_sites(xa, ya, za, xb, yb, zb)
                    extra_charge_sites += s
                    self.Nextra[s] = c.density
                    self.Sextra[s] = c.S
                    if nz == 1: # meaning not a 3D problem
                        self.Nextra[s] = self.Nextra[s] / dl
                        self.Sextra[s] = c.Sextra[s] / dl
                    self.nextra[s] = self.Nc[s] * np.exp(-self.Eg[s]/2 + c.EGB)
                    self.pextra[s] = self.Nv[s] * np.exp(-self.Eg[s]/2 - c.EGB)

        # illumination
        self.g = np.zeros((nx*ny*nz,), dtype=float)
        if self.illumination != None:
            self.g = self.illumination(self.xpts)



