import numpy as np
import scipy.constants as cts
from collections import namedtuple
from sesame.utils import get_indices
from itertools import product


def get_sites(sys, location):
    # find the sites which belongs to a region
    nx, ny, nz = sys.nx, sys.ny, sys.nz
    if sys.dimension == 1:
        s = [c for c in range(nx)]
        nodes = range(nx)
        f = lambda node: location(sys.xpts[node])
        s = [c for c in filter(f, nodes)]
    if sys.dimension == 2:
        nodes = product(range(nx), range(ny))
        f = lambda node: location((sys.xpts[node[0]], 
                                   sys.ypts[node[1]]))
        s = [c[0] + c[1]*nx for c in filter(f, nodes)]
    if sys.dimension == 3:
        nodes = product(range(nx), range(ny), range(nz))
        f = lambda node: location((sys.xpts[node[0]], 
                                   sys.ypts[node[1]], 
                                   sys.zpts[node[2]]))
        s = [c[0] + c[1]*nx + c[2]*nx*ny for c in filter(f, nodes)]
    return s

def get_line_defects_sites(system, line_defects):
    c = line_defects
    xa, ya = c.location[0]
    xb, yb = c.location[1]
    # put end points of the line in ascending order
    if ya <= yb:
        ia, ja, _ = get_indices(system, (xa, ya, 0))
        ib, jb, _ = get_indices(system, (xb, yb, 0))
    else:
        ia, ja, _ = get_indices(system, (xb, yb, 0))
        ib, jb, _ = get_indices(system, (xa, ya, 0))
        
    # find the sites closest to the straight line defined by
    # (xa,ya) and (xb,yb) and the associated orthogonal dl       
    distance = lambda x, y:\
        abs((yb-ya)*x - (xb-xa)*y + xb*ya - yb*xa)/\
            np.sqrt((yb-ya)**2 + (xb-xa)**2)

    xpts, ypts = system.xpts, system.ypts
    dx, dy = system.dx, system.dy
    nx, ny = system.nx, system.ny

    s = [ia + ja*nx]
    dl = []
    i, j = ia, ja
    def condition(i, j):
        if ia <= ib:
            return i <= ib and j <= jb and i < nx-1 and j < ny-1
        else:
            return i >= ib and j <= jb and i > 1 and j < ny-1
            
    while condition(i, j):
        # distance between the point above (i,j) and the segment
        d1 = distance(xpts[i], ypts[j+1])
        # distance between the point right of (i,j) and the segment
        d2 = distance(xpts[i+1], ypts[j])
        # distance between the point left of (i,j) and the segment
        d3 = distance(xpts[i-1], ypts[j])
        
        if ia < ib: # overall direction is to the right
            if d1 < d2:
                i, j = i, j+1
                # set dl for the previous node
                dl.append((dx[i] + dx[i-1])/2.)
            else:
                i, j = i+1, j
                # set dl for the previous node
                dl.append((dy[j] + dy[j-1])/2.)
        else: # overall direction is to the left
            if d1 < d3:
                i, j = i, j+1
                # set dl for the previous node
                dl.append((dx[i] + dx[i-1])/2.)
            else:
                i, j = i-1, j
                # set dl for the previous node
                dl.append((dy[j] + dy[j-1])/2.)
        s.append(i + j*nx)
    dl.append(dl[-1])
    dl = np.asarray(dl)
    return s, dl

def get_plane_defects_sites(system, plane_defects):
    c = plane_defects
    #TODO finish the function

    xa, ya, za = c.location[0]
    xb, yb, zb = c.location[1]
    xc, yc, zc = c.location[2]
    xd, yd, zd = c.location[3]



class Builder():
    """
    A system discretized on a mesh. 

    This type discretizes a system on a mesh provided by the user, and takes
    care of all normalizations. The temperature of the system is specified when
    an instance is created. The default is 300 K. 

    Attributes
    ----------
    xpts, ypts, zpts: numpy arrays of floats
        Mesh with original dimensions.
    dx, dy, dz: numpy arrays of floats
        Dimensionless lattice constants in the x, y, z directions.
    nx, ny, nz: integers
        Number of lattice nodes in the x, y, z directions.
    Nc, Nv: numpy arrays of floats
        Dimensionless effective densities of states of the conduction and
        valence bands.
    Eg: numpy array of floats
        Dimensionless band gap.
    mu_e, mu_h:  numpy arrays of floats
        Dimensionless mobilities of electron and holes.
    tau_e, tau_h:  numpy arrays of floats
        Dimensionless bulk lifetime for electrons and holes.
    n1, p1:  numpy arrays of floats
        Dimensionless equilibrium densities of electrons and holes at the bulk trap state.
    bl: numpy array of floats
        Dimensionless Band offset.
    g: numpy array of floats
        Dimensionless generation for each site of the
        system. This is defined only if a generation profile was provided when
        building the system.
    Nextra: numpy array of floats
        Dimensionless density of defect states. The shape of the array is (c,
        nx*ny*nz) where c is the number of added defects.
    Seextra: numpy array of floats
        Dimensionless electron recombination velocity of defect states. The
        shape of the array is (c, nx*ny*nz) where c is the number of added
        defects.
    Shextra: numpy array of floats
        Dimensionless hole recombination velocity of defect states. The shape of
        the array is (c, nx*ny*nz) where c is the number of added defects.
    nextra: numpy array of floats
        Dimensionless equilibrium electron density from the defect states. The
        shape of the array is (c, nx*ny*nz) where c is the number of added
        defects.
    pextra: numpy array of floats
        Dimensionless hole density from the defect states. The shape of the
        array is (c, nx*ny*nz) where c is the number of added defects.
    extra_charge_sites: list of lists
        List of the lists of all defect sites in the order they were added to
        the system.
    ni: numpy array of floats
        Dimensionless intrinsic density.
    scaling: named tuple
        Contains the scaling applied to physical quantities. The field names are
        density, energy, mobility, time, length, generation, velocity.
    """

    # named tuple of all the dimensions
    dimensions = namedtuple('dimensions', 
                 ['density', 'energy', 'mobility', 'time',\
                  'length', 'generation', 'velocity'])

    def __init__(self, xpts, ypts=None, zpts=None, T=300):
        # T is temperature in Kelvin

        # scalings for...
        # densities
        N = 1e19 * 1e6 # [m^-3]
        # energies
        vt = cts.k * T / cts.e
        # mobilities [m^2 / (V.s)]
        mu = 1 * 1e-4
        # time [s]
        t0 = cts.epsilon_0 / (mu * cts.e* N)
        # lengths [m]
        xscale = np.sqrt(cts.epsilon_0 * vt / (cts.e * N))
        # generation rate [m^-3]
        U = (N * mu * vt) / xscale**2 
        # recombination velocities
        Sc = xscale / t0
        self.scaling = self.dimensions(N, vt, mu, t0, xscale, U, Sc)


        self.xpts = xpts
        self.dx = (self.xpts[1:] - self.xpts[:-1]) / xscale
        self.nx = xpts.shape[0]
        self.dimension = 1

        self.ypts = ypts
        self.ny = 1
        if ypts is not None:
            self.ypts = ypts
            self.dy = (self.ypts[1:] - self.ypts[:-1]) / xscale
            self.ny = ypts.shape[0]
            self.dimension = 2

        self.zpts = zpts
        self.nz = 1
        if zpts is not None:
            self.zpts = zpts
            self.dz = (self.zpts[1:] - self.zpts[:-1]) / xscale
            self.nz = zpts.shape[0]
            self.dimension = 3

        nx, ny, nz = self.nx, self.ny, self.nz
        self.Nc      = np.zeros((nx*ny*nz,), dtype=float)
        self.Nv      = np.zeros((nx*ny*nz,), dtype=float)
        self.Eg      = np.zeros((nx*ny*nz,), dtype=float)
        self.epsilon = np.zeros((nx*ny*nz,), dtype=float)
        self.mu_e    = np.zeros((nx*ny*nz,), dtype=float)
        self.mu_h    = np.zeros((nx*ny*nz,), dtype=float)
        self.tau_e   = np.zeros((nx*ny*nz,), dtype=float)
        self.tau_h   = np.zeros((nx*ny*nz,), dtype=float)
        self.n1      = np.zeros((nx*ny*nz,), dtype=float)
        self.p1      = np.zeros((nx*ny*nz,), dtype=float)
        self.bl      = np.zeros((nx*ny*nz,), dtype=float)
        self.rho     = np.zeros((nx*ny*nz,), dtype=float)
        self.g       = np.zeros((nx*ny*nz,), dtype=float)


        # list of lines defects
        self.line_defects = namedtuple('line_defects', \
        ['location', 'energy', 'density', 'Se', 'Sh'])
        self.lines_defects = []

        # list of planes defects
        self.plane_defects = namedtuple('plane_defects', \
        ['location', 'energy', 'density', 'Se', 'Sh'])
        self.planes_defects = []


    def add_material(self, mat, location=lambda pos: True):
        """
        Add a material to the system.

        Parameters
        ----------
        mat: dictionary 
            Contains the material parameters
            Keys are Nc (Nv): conduction (valence) effective densities of
            states [m\ :sup:`-3`], Eg: band gap [:math:`\mathrm{eV}`], epsilon: material's
            permitivitty, mu_e (mu_h): electron (hole) mobility
            [m\ :sup:`2`/V/s],
            tau_e (tau_h): electron (hole) bulk lifetime [s], RCenergy: energy
            level of the bulk recombination centers [eV], band_offset: band
            offset setting the zero of potential [eV].
        location: Boolean function
            Definition of the region containing the material. This function must
            take a tuple of real world coordinates (e.g. (x, y)) as parameters, and return True (False) if the
            lattice node is inside (outside) the region.
        """

        # sites belonging to the region
        s = get_sites(self, location)

        # fill in arrays
        self.Nc[s]      = mat['Nc'] / self.scaling.density
        self.Nv[s]      = mat['Nv'] / self.scaling.density
        self.Eg[s]      = mat['Eg'] / self.scaling.energy
        self.epsilon[s] = mat['epsilon']
        self.mu_e[s]    = mat['mu_e'] / self.scaling.mobility
        self.mu_h[s]    = mat['mu_h'] / self.scaling.mobility
        self.tau_e[s]   = mat['tau_e'] / self.scaling.time
        self.tau_h[s]   = mat['tau_h'] / self.scaling.time
        self.bl[s]      = mat['band_offset'] / self.scaling.energy

        Etrap = mat['RCenergy'] / self.scaling.energy
        self.n1[s]      = self.Nc[s] * np.exp(-self.Eg[s]/2 + Etrap)
        self.p1[s]      = self.Nv[s] * np.exp(-self.Eg[s]/2 - Etrap)


    def add_line_defects(self, location, local_E, local_N, local_Se,\
                          local_Sh=None):
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system.

        Parameters
        ----------
        location: list of two (x, y) tuples 
            The coordinates in [m] define a line of defects in 2D.
        local_E: float 
            Energy level of the states defined with respect to E\ :sub:`g`/2 [eV].
        local_N: float
            Defect density of states [m\ :sup:`-2` ].
        local_Se: float
            Surface recombination velocity of electrons [m/s].
        local_Sh: float
            Surface recombination velocity of holes [m/s].

        Warnings
        --------
        * Addition of line defects is defined for two-dimensional systems only.

        * We assume that no additional charge is on the contacts.

        See Also
        --------
        add_plane_defects

        """
        
        # if one wants same S for electrons and holes
        if local_Sh == None:
            local_Sh = local_Se

        d = self.line_defects(location, local_E / self.scaling.energy, \
                    local_N / (self.scaling.density * self.scaling.length), \
                    local_Se / self.scaling.velocity, 
                    local_Sh / self.scaling.velocity)
        self.lines_defects.append(d)

    def add_plane_defects(self, location, local_E, local_N, local_Se,\
                          local_Sh=None):
        
        # if one wants same S for electrons and holes
        if local_Sh == None:
            local_Sh = local_Se

        d = self.line_defects(location, local_E / self.scaling.energy, \
                    local_N / (self.scaling.density * self.scaling.length), \
                    local_Se / self.scaling.velocity, 
                    local_Sh / self.scaling.velocity)

        self.planes_defects.append(d)



    def doping_profile(self, density, location=lambda pos: True):
        """
        Add dopant charges to the system.

        Parameters
        ----------
        density: float
            Doping density [m\ :sup:`-3`].
        location: Boolean function
            Definition of the region containing the doping. This function must
            take a tuple of real world coordinates (e.g. (x, y)) as parameters,
            and return True (False) if the lattice node is inside (outside) the
            region.
        """

        s = get_sites(self, location)
        self.rho[s] = density / self.scaling.density

    def add_donor(self, density, location):
        self.doping_profile(density, location)

    def add_acceptor(self, density, location):
        self.doping_profile(-density, location)

    def generation(self, f):
        """
        Distribution of photogenerated carriers.

        Parameters
        ----------
        f: function 
            Generation rate [m\ :sup:`-3`].
        """
        if self.dimension == 1:
            g = [f(x) for x in self.xpts]
        elif self.dimension == 2:
            g = [f(x, y) for y in self.ypts for x in self.xpts]
        elif self.dimension == 3:
            g = [f(x, y, z) for z in self.zpts for y in self.ypts for x in self.xpts]
        self.g = np.asarray(g) / self.scaling.generation


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
        Use 10\ :sup:`50` for infinite surface recombination velocities.
        """

        self.Scn = [Scn_left / self.scaling.velocity, 
                    Scn_right / self.scaling.velocity]
        self.Scp = [Scp_left / self.scaling.velocity, 
                    Scp_right / self.scaling.velocity]


    def finalize(self):
        """
        Generate the arrays containing the quantities related to the charges
        added to the system (i.e. line defects or plane defects).
        """

        # mesh parameters
        nx, ny, nz = self.nx, self.ny, self.nz

        # intrinsic density across the entire system
        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg/2)

        # additional extra charges
        c = len(self.lines_defects) + len(self.planes_defects)
        if c != 0:
            self.Nextra  = np.zeros((c, nx*ny*nz), dtype=float)
            self.Seextra = np.zeros((c, nx*ny*nz), dtype=float)
            self.Shextra = np.zeros((c, nx*ny*nz), dtype=float)
            self.nextra  = np.zeros((c, nx*ny*nz), dtype=float)
            self.pextra  = np.zeros((c, nx*ny*nz), dtype=float)
            self.extra_charge_sites = []

            # fill in charges from lines defects
            for cdx, c in enumerate(self.lines_defects):
                s, dl = get_line_defects_sites(self, c)

                self.extra_charge_sites += [s]

                self.Nextra[cdx, s]  = self.Nextra[cdx, s] / dl
                self.Seextra[cdx, s] = self.Seextra[cdx, s] / dl
                self.Shextra[cdx, s] = self.Shextra[cdx, s] / dl
                self.nextra[cdx, s] = self.Nc[s] * np.exp(-self.Eg[s]/2 + c.energy)
                self.pextra[cdx, s] = self.Nv[s] * np.exp(-self.Eg[s]/2 - c.energy)
            # fill in charges from plane defects
            for cdx, c in enumerate(self.planes_defects):
                s = get_plane_defects_sites(self, c)

                self.extra_charge_sites += [s]

                self.Nextra[cdx, s]  = c.density
                self.Seextra[cdx, s] = c.Se
                self.Shextra[cdx, s] = c.Sh



