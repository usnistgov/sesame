import numpy as np
import scipy.constants as cts
from collections import namedtuple
from itertools import product
from . import utils


def get_sites(sys, location):
    # find the sites which belongs to a region
    nx, ny, nz = sys.nx, sys.ny, sys.nz
    if sys.dimension == 1:
        nodes = range(nx)
        s = [i for i in range(nx) if location(sys.xpts[i])]
    if sys.dimension == 2:
        nodes = product(range(nx), range(ny))
        s = [i + j*nx for i, j in nodes if location((sys.xpts[i], sys.ypts[j]))]
    if sys.dimension == 3:
        nodes = product(range(nx), range(ny), range(nz))
        s = [i + j*nx + k*nx*ny for i, j, k in nodes if location((sys.xpts[i], 
                                                                  sys.ypts[j], 
                                                                  sys.zpts[k])
                                                                )
            ]
    return s

def get_line_defects_sites(system, line_defects):
    # find the sites closest to the straight line defined by
    # (xa,ya,za) and (xb,yb,zb) 

    xa, ya = line_defects.location[0]
    xb, yb = line_defects.location[1]
    ia, ja, _ = utils.get_indices(system, (xa, ya, 0))
    ib, jb, _ = utils.get_indices(system, (xb, yb, 0))

    Dx = abs(ib - ia)    # distance to travel in X
    Dy = abs(jb - ja)    # distance to travel in Y
    if ia < ib:
        incx = 1           # x will increase at each step
    elif ia > ib:
        incx = -1          # x will decrease at each step
    else:
        incx = 0
    if ja < jb:
        incy = 1           # y will increase at each step
    elif ja > jb:
        incy = -1          # y will decrease at each step
    else:
        incy = 0

    # take the numerator of the distance of a point to the line
    error = lambda x, y: abs((yb-ya)*x - (xb-xa)*y + xb*ya - yb*xa)

    i, j = ia, ja
    perp_dl = []
    sites = [i + j*system.nx]
    for _ in range(Dx + Dy):
        e1 = error(system.xpts[i], system.ypts[j+incy])
        e2 = error(system.xpts[i+incx], system.ypts[j])
        if e1 < e2:
            j += incy
            perp_dl.append((system.dx[i] + system.dx[i-1])/2.)  
        else:
            i += incx
            perp_dl.append((system.dy[j] + system.dy[j-1])/2.)
        sites.append(i + j*system.nx)
    perp_dl.append(perp_dl[-1])
    perp_dl = np.asarray(perp_dl)

    return sites, perp_dl



def get_plane_defects_sites(system, plane_defects):
    # This will only work for planes parallel to at least one direction. These
    # planes should be defined by two parallel lines orthogonal to the z-axis,
    # and be rectangles.

    # first line
    P1 = np.asarray(plane_defects.location[0])
    P2 = np.asarray(plane_defects.location[1])
    # second line
    P3 = np.asarray(plane_defects.location[2])
    P4 = np.asarray(plane_defects.location[3])

    sites, _, _, _ = utils.extra_charges_plane(system, P1, P2, P3, P4) 

    return sites        



class Builder():
    """
    A system discretized on a mesh. 

    This type discretizes a system on a mesh provided by the user, and takes
    care of all normalizations. The temperature of the system is specified when
    an instance is created. The default is 300 K. 

    Parameters
    ----------
    xpts, ypts, zpts: numpy arrays of floats
        Mesh with original dimensions.


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

        # named tuple of all the dimensions
        dimensions = namedtuple('dimensions', 
                     ['density', 'energy', 'mobility', 'time',\
                      'length', 'generation', 'velocity'])
        self.scaling = dimensions(N, vt, mu, t0, xscale, U, Sc)


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
        self.ni      = np.zeros((nx*ny*nz,), dtype=float)


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

        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg/2)

    def add_line_defects(self, location, local_E, local_N, local_Se,\
                          local_Sh=None):
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system. These charges are distributed on a line.

        Parameters
        ----------
        location: list of two array_like coordinates [(x1, y1), (x2, y2)] 
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
        add_plane_defects for adding plane defects in 3D.
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
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system. These charges are distributed on a plane.

        Parameters
        ----------
        location: list of four array_like coordinates [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)] 
            The coordinates in [m] define a plane of defects in 3D. The first
            two coordinates define a line that must be parallel to the line
            defined by the last two points.
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
        * The planes must be rectangles with at least one edge parallel to
          either the x or y or z-axis.

        * Addition of plane defects is defined for three-dimensional systems only.

        * We assume that no additional charge is on the contacts.

        See Also
        --------
        add_line_defects for adding line defects in 2D.
        """

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

                self.Nextra[cdx, s]  = c.density / dl
                self.Seextra[cdx, s] = c.Se / dl
                self.Shextra[cdx, s] = c.Sh / dl
                self.nextra[cdx, s] = self.Nc[s] * np.exp(-self.Eg[s]/2 + c.energy)
                self.pextra[cdx, s] = self.Nv[s] * np.exp(-self.Eg[s]/2 - c.energy)

            # fill in charges from plane defects
            for cdx, c in enumerate(self.planes_defects):
                s = get_plane_defects_sites(self, c)

                self.extra_charge_sites += [s]

                self.Nextra[cdx, s]  = c.density
                self.Seextra[cdx, s] = c.Se
                self.Shextra[cdx, s] = c.Sh
                self.nextra[cdx, s] = self.Nc[s] * np.exp(-self.Eg[s]/2 + c.energy)
                self.pextra[cdx, s] = self.Nv[s] * np.exp(-self.Eg[s]/2 - c.energy)



