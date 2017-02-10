from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
import scipy.constants as cts
from collections import namedtuple
from itertools import product

from . import utils

# named tuple of all the dimensions
dimensions = namedtuple('dimensions', ['density', 'energy', 'mobility', 'time',\
                                       'length', 'generation', 'velocity'])

# named tuple of the characteristics of a defect
defect = namedtuple('defect', ['sites', 'location', \
                               'dos', 'energy', 'sigma_e', 'sigma_h',\
                               'transition', 'perp_dl'])

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
    T: float
        Temperature for the simulation.


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
    ni: numpy array of floats
        Dimensionless intrinsic density.
    mu_e, mu_h:  numpy arrays of floats
        Dimensionless mobilities of electron and holes.
    tau_e, tau_h:  numpy arrays of floats
        Dimensionless bulk lifetime for electrons and holes.
    n1, p1:  numpy arrays of floats
        Dimensionless equilibrium densities of electrons and holes at the bulk trap state.
    bl: numpy array of floats
        Dimensionless band offset.
    g: numpy array of floats
        Dimensionless generation for each site of the
        system. This is defined only if a generation profile was provided when
        building the system.
    gtot: float
        Dimensionless integral of the generation rate.
    defects_list: list of named tuples
        List of named tuples containing the characteristics ofthe defects in the
        order they were added to the system. The field names are sites,
        location, dos, energy, sigma_e, sigma_h, transition, perp_dl.
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
        # generation rate [m^-3 s^-1]
        U = (N * mu * vt) / xscale**2 
        # recombination velocities
        Sc = xscale / t0

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
        self.mass_e  = np.zeros((nx*ny*nz,), dtype=float)
        self.mass_h  = np.zeros((nx*ny*nz,), dtype=float)
        self.mu_e    = np.zeros((nx*ny*nz,), dtype=float)
        self.mu_h    = np.zeros((nx*ny*nz,), dtype=float)
        self.tau_e   = np.zeros((nx*ny*nz,), dtype=float)
        self.tau_h   = np.zeros((nx*ny*nz,), dtype=float)
        self.n1      = np.zeros((nx*ny*nz,), dtype=float)
        self.p1      = np.zeros((nx*ny*nz,), dtype=float)
        self.bl      = np.zeros((nx*ny*nz,), dtype=float)
        self.rho     = np.zeros((nx*ny*nz,), dtype=float)
        self.g       = np.zeros((nx*ny*nz,), dtype=float)
        self.B       = np.zeros((nx*ny*nz,), dtype=float)
        self.Cn      = np.zeros((nx*ny*nz,), dtype=float)
        self.Cp      = np.zeros((nx*ny*nz,), dtype=float)

        self.defects_list = []

    def add_material(self, mat, location=lambda pos: True):
        """
        Add a material to the system.

        Parameters
        ----------
        mat: dictionary 
            Contains the material parameters
            Keys are Nc (Nv): conduction (valence) effective densities of states
            [m\ :sup:`-3`], Eg: band gap [:math:`\mathrm{eV}`], epsilon:
            material's permitivitty, mu_e (mu_h): electron (hole) mobility [m\
            :sup:`2`/V/s], tau_e (tau_h): electron (hole) bulk lifetime [s], Et:
            energy level of the bulk recombination centers [eV], band_offset:
            band offset setting the zero of potential [eV], B: radiation
            recombination constant [m\ :sup:`3`/s], Cn (Cp): Auger recombination constant for
            electrons (holes) [m\ :sup:`6`/s], mass_e (mass_h): effective mass of electrons
            (holes).
        location: Boolean function
            Definition of the region containing the material. This function must
            take a tuple of real world coordinates (e.g. (x, y)) as parameters,
            and return True (False) if the lattice node is inside (outside) the
            region.
        """

        # sites belonging to the region
        s = get_sites(self, location)

        N = self.scaling.density
        t = self.scaling.time
        vt = self.scaling.energy
        mu = self.scaling.mobility

        # default material parameters
        mt = {'Nc': 1e25, 'Nv': 1e25, 'Eg': 1, 'epsilon': 1, 'mass_e': 1,\
              'mass_h': 1, 'mu_e': 100, 'mu_h': 100, 'Et': 0, 'tau_e': 1e-6,\
              'tau_h': 1e-6, 'band_offset': 0, 'B': 0, 'Cn': 0, 'Cp': 0}

        for key in mat.keys():
            mt[key] = mat[key]

        # fill in arrays
        self.Nc[s]      = mt['Nc'] / N
        self.Nv[s]      = mt['Nv'] / N
        self.Eg[s]      = mt['Eg'] / vt
        self.epsilon[s] = mt['epsilon']
        self.mass_e[s]  = mt['mass_e']
        self.mass_h[s]  = mt['mass_h']
        self.mu_e[s]    = mt['mu_e'] / mu
        self.mu_h[s]    = mt['mu_h'] / mu
        self.tau_e[s]   = mt['tau_e'] / t
        self.tau_h[s]   = mt['tau_h'] / t
        self.bl[s]      = mt['band_offset'] / vt
        self.B[s]       = mt['B'] / ((1./N)/t)
        self.Cn[s]      = mt['Cn'] / ((1./N**2)/t)
        self.Cp[s]      = mt['Cp'] / ((1./N**2)/t)

        Etrap = mt['Et'] / self.scaling.energy
        self.n1[s]      = self.Nc[s] * np.exp(-self.Eg[s]/2 + Etrap)
        self.p1[s]      = self.Nv[s] * np.exp(-self.Eg[s]/2 - Etrap)

        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg/2)

    def add_defects(self, location, N, sigma_e, sigma_h, E, transition):

        if E is not None:
            E /= self.scaling.energy

        if len(location) == 2:
            s, dl = get_line_defects_sites(self, location)
        elif len(location) == 4:
            s, _, _, _ = utils.plane_defects_sites(self, location) 
            dl = 1.
        else:
            print("Wrong definition for the defects location: the list must"+\
                  "contain two points for a line, four points for a plane.")
            exit(1)


        if sigma_h == None:
            sigma_h = sigma_e
        sigma_e /= self.scaling.length**2
        sigma_h /= self.scaling.length**2

        NN = self.scaling.density * self.scaling.length
        if isinstance(N, float):
            f = N / NN
        else:
            f = lambda E: N(E*self.scaling.energy) / NN

        params = defect(s, location, f, E, sigma_e, sigma_h, transition, dl)
        self.defects_list.append(params)
 
    def add_line_defects(self, location, N, Se, Sh=None, E=None, transition=(1,-1)):
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system. These charges are distributed on a line.

        Parameters
        ----------
        location: list of two array_like coordinates [(x1, y1), (x2, y2)] 
            The coordinates in [m] define a line of defects in 2D.
        N: float or function
            Defect density of states [m\ :sup:`-2` ]. Provide a float when the
            defect density of states is a delta function, or a function
            returning a float for a continuum. This function should take a
            single energy argument in [eV].
        Se: float
            Surface recombination velocity of electrons [m/s].
        Sh: float
            Surface recombination velocity of holes [m/s].
        E: float 
            Energy level of a single state defined with respect to E\ :sub:`g`/2
            [eV]. Set to `None` for a continuum of states.
        transition: tuple
            Charge transition occurring at the energy level E.  The tuple (p, q)
            represents a defect with transition p/q (level empty to level
            occupied).

        Warnings
        --------
        * Addition of line defects is defined for two-dimensional systems only.

        * We assume that no additional charge is on the contacts.

        See Also
        --------
        add_plane_defects
        """
           
        self.add_defects(location, N, Se, Sh, E, transition)

    def add_plane_defects(self, location, N, Se, Sh=None, E=None, transition=(1,-1)):
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system. These charges are distributed on a plane.

        Parameters
        ----------
        location: list of four array_like coordinates [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)] 
            The coordinates in [m] define a plane of defects in 3D. The first
            two coordinates define a line that must be parallel to the line
            defined by the last two points.
        N: float or function
            Defect density of states [m\ :sup:`-2` ]. Provide a float when the
            defect density of states is a delta function, or a function
            returning a float for a continuum. This function should take a
            single energy argument in [eV].
        Se: float
            Surface recombination velocity of electrons [m/s].
        Sh: float
            Surface recombination velocity of holes [m/s].
        E: float 
            Energy level of a single state defined with respect to E\ :sub:`g`/2
            [eV]. Set to `None` for a continuum of states.
        transition: tuple
            Charge transition occurring at the energy level E.  The tuple (p, q)
            represents a defect with transition p/q (level empty to level
            occupied).

        Warnings
        --------
        * The planes must be rectangles with at least one edge parallel to
          either the x or y or z-axis.

        * Addition of plane defects is defined for three-dimensional systems only.

        * We assume that no additional charge is on the contacts.

        See Also
        --------
        add_line_defects
        """
        self.add_defects(location, N, Se, Sh, E, transition)
 
    def doping_profile(self, density, location):

        s = get_sites(self, location)
        self.rho[s] = density / self.scaling.density

    def add_donor(self, density, location=lambda pos: True):
        """
        Add donor dopants to the system.

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
        self.doping_profile(density, location)

    def add_acceptor(self, density, location=lambda pos: True):
        """
        Add acceptor dopants to the system.

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
        
        # compute the integral of the generation
        x = self.xpts / self.scaling.length
        if self.ny > 1:
            y = self.ypts / self.scaling.length
        if self.nz > 1: 
            z = self.zpts / self.scaling.length

        w = []
        for k in range(self.nz):
            u = []
            for j in range(self.ny):
                s = [i + j*self.nx + k*self.nx*self.ny  for i in range(self.nx)]
                sp = spline(x, self.g[s])
                u.append(sp.integral(x[0], x[-1]))
            if self.dimension > 1:
                sp = spline(y, u)
                w.append(sp.integral(y[0], y[-1]))
        if self.dimension == 1:
            self.gtot = u[-1]
        if self.dimension == 2:
            self.gtot = w[-1]
        if self.dimension == 3:
            sp = spline(z, u)
            self.gtot = sp.integral(z[0], z[-1])
 

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


def get_sites(sys, location):
    # find the sites which belongs to a region
    nx, ny, nz = sys.nx, sys.ny, sys.nz
    sites = np.arange(nx*ny*nz, dtype=int)
 
    if sys.dimension == 1:
        mask = location((sys.xpts))

    if sys.dimension == 2:
        pos = np.transpose([np.tile(sys.xpts, ny), np.repeat(sys.ypts, nx)])
        mask = location((pos[:,0], pos[:,1]))

    if sys.dimension == 3:
        pos = np.reshape(np.concatenate((np.tile(sys.xpts, ny*nz),
                                         np.repeat(sys.ypts, nx*nz),
                                         np.repeat(sys.zpts, nx*ny)
                                        )
                                       ),
                         (3, nx*ny*nz)
                        ).T
        mask = location((pos[:,0], pos[:,1], pos[:,2]))

    if type(mask) == bool:
        return sites
    else:
        return sites[mask.astype(bool)]


def get_line_defects_sites(system, location):
    # find the sites closest to the straight line defined by
    # (xa,ya,za) and (xb,yb,zb) 

    xa, ya = location[0]
    xb, yb = location[1]
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
