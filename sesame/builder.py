import numpy as np
import scipy.constants as cts
from collections import namedtuple
from sesame.utils import get_indices, get_xyz_from_s

class Builder():
    """
    A system discretized on a mesh. 

    This type discretizes a system on a mesh provided by the user, and takes
    care of all normalizations. The temperature of the system is specified when
    an instance is created. The default is 300 K. 

    Attributes
    ----------
    N: 10^25 m⁻3
        Density scale.
    vt: kT/e
        Thermal velocity (voltage scale) [V].
    mu: :math`10^-4 m^2/(V\cdot s)`
        Mobility scale.
    t0: `\epsilon_0 vt / (e N)`
        Time scale [s].
    xscale: :math:`\sqrt{\epsilon_0 vt/(eN)}`
        Length scale [m].
    U: :math:`N\ mu\  vt/ xscale^2`
        Generation rate scale [m^-3].
    Sc: xscale/t0
        Surface recombination velocity scale [m/s].
    """

    def __init__(self, T=300):
    # temperature in Kelvin
        self.T = T

        # scalings for...
        # densities
        self.N = 1e19 * 1e6 # [m^-3]
        # NN = self.N
        # energies
        self.vt = cts.k * T / cts.e
        # mobilities [m^2 / (V.s)]
        self.mu = 1 * 1e-4
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
                                            'energy', 'density', 'Se', 'Sh'])
        self.charges = []

        # generation of carriers
        self.g = 0 # no illumination by default 
        # length of mesh in x, y, z directions
        self.nx, self.ny, self.nz = 1, 1, 1
        self.dimension = 1



    def add_material(self, location, mat):
        """
        Add a material to the system.

        Parameters
        ----------
        location: list of two (x, y, z) tuples 
            The coordinates define in [m] define the material's region (a
            rectangle in 2D). Use zeros for unused dimensions.
        mat: dictionary containing the material parameters
            Keys are Nc (Nv): conduction (valence) effective densities of
            states [1/m^3], Eg: band gap [eV], epsilon: material's
            permitivitty, mu_e (mu_h): electron (hole) mobility [m^2/(V.s)],
            tau_e (tau_h): electron (hole) bulk lifetime, RCenergy: energy
            level of the bulk recombination centers [eV].
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

    def add_local_charges(self, location, local_E, local_N, local_Se,\
                          local_Sh=None):
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system.

        Parameters
        ----------
        location: list of two (x, y, z) tuples 
            The coordinates in [m] define a line of defect in 2D or a plane in
            3D. Use zeros for unused dimensions.
        local_E: float 
            Energy level of the states defined with respect to Eg/2 [eV].
        local_N: float
            Defect density of states [m^-2].
        local_Se: float
            Surface recombination velocity of electrons [m.s^-1].
        local_Sh: float
            Surface recombination velocity of holes [m.s^-1].

        Warnings
        --------
        * In 2D the two points defining a line of defects must be given in
        ascending order (lowest y-coordinate first).

        * We assume that no additional charge is on the contacts.

        """
        
        xa, ya, za = location[0]
        xb, yb, zb = location[1]
        
        # if one wants same S for electrons and holes
        if local_Sh == None:
            local_Sh = local_Se

        d = self.charge(xa/self.xscale, ya/self.xscale, za/self.xscale,
                        xb/self.xscale, yb/self.xscale, zb/self.xscale,
                        local_E/self.vt, local_N/(self.N*self.xscale), 
                        local_Se/self.Sc, local_Sh/self.Sc)
        self.charges.append(d)

    def doping_profile(self, location, density):
        """
        Add dopant charges to the system.

        Parameters
        ----------
        location: list of two (x, y, z) tuples 
            The coordinates in [m] define the region of doping (a rectangle in
            2D). Use zeros for unused dimensions.
        density: float
            Doping density [m^-3].
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

    def illumination(self, f, scale=True):
        """
        Distribution of photogenerated carriers.

        Parameters
        ----------
        f: function 
            Generation rate [m^-3].
        scale: boolean 
            Determines if scaling should be applied (True) or not (False).
        """

        if scale:
            self.illumination = lambda x, y, z: f(x, y, z) / self.U
        else:
            self.illumination = lambda x, y, z: f(x, y, z)
        self.g = 1

    def generation(self, g):
        self.g = g
        

    def contacts(self, Scn_left, Scp_left, Scn_right, Scp_right):
        """
        Create the lists of recombination velocities that define the contacts
        boundary conditions.

        Parameters
        ----------
        Scn_left: float
            Surface recombination velocity for electrons at the left contact [m/s].
        Scp_left: float
            Surface recombination velocity for holes at the left contact [m/s].
        Scn_right: float
            Surface recombination velocity for electrons at the right contact [m/s].
        Scn_right: float
            Surface recombination velocity for electrons at the right contact [m/s].

        Notes
        -----
        Use 10^50 for infinite surface recombination velocities.
        """

        self.Scn = [Scn_left/self.Sc, Scn_right/self.Sc]
        self.Scp = [Scp_left/self.Sc, Scp_right/self.Sc]

    def mesh(self, xpts, ypts=None, zpts=None):
        """
        Mesh of the system.

        Parameters
        ----------
        xpts: numpy array
            Mesh points in the x-direction in [m].
        ypts: numpy array
            Mesh points in the y-direction in [m].
        zpts: numpy array
            Mesh points in the z-direction in [m].
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
        Generate the arrays containing all the parameters of the system.
        """

        # mesh parameters
        nx, ny, nz = self.nx, self.ny, self.nz

        def get_sites(xa, ya, za, xb, yb, zb):
            if yb == self.ny: yb = self.ny-1
            if zb == self.nz: yb = self.nz-1
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
            self.rho[s] = d.density # divided by epsilon later

        # additional extra charges
        #!!! DO NOT add extra charges on the contact sites

        if len(self.charges) != 0:
            self.Nextra = np.zeros((len(self.charges), nx*ny*nz), dtype=float)
            self.Seextra = np.zeros((len(self.charges), nx*ny*nz), dtype=float)
            self.Shextra = np.zeros((len(self.charges), nx*ny*nz), dtype=float)
            self.nextra = np.zeros((len(self.charges), nx*ny*nz), dtype=float)
            self.pextra = np.zeros((len(self.charges), nx*ny*nz), dtype=float)
            self.extra_charge_sites = []
            for cdx, c in enumerate(self.charges):
                xa, ya, za = get_indices(self, (c.xa, c.ya, c.za))
                xb, yb, zb = get_indices(self, (c.xb, c.yb, c.zb))

                # find the sites closest to the straight line defined by
                # (xa,ya,za) and (xb,yb,zb) and the associated dl       

                # XXX TODO generalize to 3D and the distance to a plane
                distance = lambda x, y:\
                    abs((c.yb-c.ya)*x - (c.xb-c.xa)*y + c.xb*c.ya - c.yb*c.xa)/\
                        np.sqrt((c.yb-c.ya)**2 + (c.xb-c.xa)**2)

                s = [xa + ya*nx]
                dl = []
                x, y = xa, ya
                def condition(x, y):
                    if xa <= xb:
                        return x <= xb and y <= yb and x < nx-1 and y < ny-1
                    else:
                        return x >= xb and y <= yb and x > 1 and y < ny-1
                        
                while condition(x, y):
                    # distance between the point above (x,y) and the segment
                    d1 = distance(self.xpts[x], self.ypts[y+1])
                    # distance between the point right of (x,y) and the segment
                    d2 = distance(self.xpts[x+1], self.ypts[y])
                    # distance between the point left of (x,y) and the segment
                    d3 = distance(self.xpts[x-1], self.ypts[y])
                    
                    if xa < xb: # overall direction is to the right
                        if d1 < d2:
                            x, y = x, y+1
                            # set dl for the previous node
                            dl.append((self.dx[x] + self.dx[x-1])/2.)
                        else:
                            x, y = x+1, y
                            # set dl for the previous node
                            dl.append((self.dy[y] + self.dy[y-1])/2.)
                    else: # overall direction is to the left
                        if d1 < d3:
                            x, y = x, y+1
                            # set dl for the previous node
                            dl.append((self.dx[x] + self.dx[x-1])/2.)
                        else:
                            x, y = x-1, y
                            # set dl for the previous node
                            dl.append((self.dy[y] + self.dy[y-1])/2.)
                    s.append(x + y*nx)
                dl.append(dl[-1])
                dl = np.asarray(dl)

                if nz > 1: # no tilted planes
                    s = get_sites(xa, ya, za, xb, yb, zb)

                # fill arrays of DOS and surface recombination velocities
                self.extra_charge_sites += [s]
                self.Nextra[cdx, s] = c.density
                self.Seextra[cdx, s] = c.Se
                self.Shextra[cdx, s] = c.Sh
                if ny > 1 and nz == 1: # meaning a 2D problem
                    self.Nextra[cdx, s] = self.Nextra[cdx, s] / dl
                    self.Seextra[cdx, s] = self.Seextra[cdx, s] / dl
                    self.Shextra[cdx, s] = self.Shextra[cdx, s] / dl
                self.nextra[cdx, s] = self.Nc[s] * np.exp(-self.Eg[s]/2 + c.energy)
                self.pextra[cdx, s] = self.Nv[s] * np.exp(-self.Eg[s]/2 - c.energy)

        # illumination
        if self.g == 0:
            self.g = np.zeros((nx*ny*nz,), dtype=float)
        else:
            if self.dimension == 1:
                g = [self.illumination(x, 0, 0) for x in self.xpts*self.xscale]
            elif self.dimension == 2:
                g = [self.illumination(x, y, 0) for y in self.ypts*self.xscale 
                                                for x in self.xpts*self.xscale]
            elif self.dimension == 3:
                g = [self.illumination(x, y, z) for z in self.zpts*self.xscale 
                                                for y in self.ypts*self.xscale
                                                for x in self.xpts*self.xscale]
            self.g = np.asarray(g)
