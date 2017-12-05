# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from scipy.interpolate import InterpolatedUnivariateSpline as spline, interp2d
from .utils import Bresenham, get_indices
from .observables import *

try:
    import matplotlib.pyplot as plt
    mpl_enabled = True
    try:
        from mpl_toolkits import mplot3d
        has3d = True
    except:
        has3d = False
except:
    mpl_enabled = False



class Analyzer():
    """
    Object that simplifies the extraction of physical data (densities, currents,
    recombination) across the system.

    Parameters
    ----------

    sys: Builder
        A discretized system.
    data: dictionary
        Dictionary containing 1D arrays of electron and hole quasi-Fermi levels
        and the electrostatic potential across the system. Keys must be 'efn',
        'efp', and/or 'v'.
    """

    def __init__(self, sys, data):
        self.sys = sys
        self.v = data['v']

        # check for efn
        if 'efn' in data.keys():
            self.efn = data['efn']
        else:
            self.efn = 0 * self.v

        # check for efp
        if 'efp' in data.keys():
            self.efp = data['efp']
        else:
            self.efp = 0 * self.v

        # sites of the system
        self.sites = np.arange(sys.nx*sys.ny*sys.nz, dtype=int)

    @staticmethod
    def line(system, p1, p2):
        """
        Compute the path and sites between two points.

        Parameters
        ----------
        system: Builder
            The discretized system.
        p1, p2: array-like (x, y)
            Two points defining a line.

        Returns
        -------
        s, sites: numpay arrays
            Curvilinear abscissa and sites of the line.

        Notes
        -----
        This method can be used with an instance of the Analyzer():

        >>> az = sesame.Analyzer(sys, res)
        >>> X, sites = az.line(sys, p1, p2)

        or without it:

        >>> X, sites = sesame.Analyzer.line(sys, p1, p2)
        """

        p1 = (p1[0], p1[1], 0)
        p2 = (p2[0], p2[1], 0)
        s, x, _, _, _ = Bresenham(system, p1, p2)
        return x, s

    def band_diagram(self, location, fig=None):
        """
        Compute the band diagram between two points defining a line. Display a
        plot if fig is None.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute a band
            diagram.

        fig: Maplotlib figure
            A plot is added to it if given. If not given, a new one is created and 
            displayed.

        """
        p1, p2 = location
        if self.sys.dimension == 1:
            idx1, _, _ = get_indices(self.sys, (p1[0],0,0))
            idx2, _, _ = get_indices(self.sys, (p2[0],0,0))
            X = self.sys.xpts[idx1:idx2]
            sites = np.arange(idx1, idx2, 1, dtype=int)
        if self.sys.dimension == 2:
            X, sites = self.line(self.sys, p1, p2)

        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        # add axis to figure
        ax = fig.add_subplot(111)

        vt = self.sys.scaling.energy
        X = X * 1e6 # in um

        l1, = ax.plot(X, vt*self.efn[sites], lw=2, color='#2e89cfff', ls='--')
        l2, = ax.plot(X, -vt*self.efp[sites], lw=2, color='#cf392eff', ls='--')

        fig.legend([l1, l2], [r'$\mathregular{E_{F_n}}$',\
                              r'$\mathregular{-E_{F_p}}$'])

        ax.set_xlabel('position (µm)')
        ax.set_ylabel('Energy (eV)')

        if show:
            plt.show()

    def electron_density(self, location=None):
        """
        Compute the electron density across the system or on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the electron
            density.
        
        Returns
        -------
        n: numpy array of floats

        See also
        --------
        hole_density

        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)
        n = get_n(self.sys, self.efn, self.v, sites)
        return n

    def hole_density(self, location=None):
        """
        Compute the hole density across the system or  on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the hole
            density.
        
        Returns
        -------
        p: numpy array of floats

        See also
        --------
        electron_density

        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)
        p = get_p(self.sys, self.efp, self.v, sites)
        return p

    def bulk_srh_rr(self, location=None):
        """
        Compute the bulk Shockley-Read-Hall recombination across the system or
        on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            TUple of two points defining a line over which to compute the recombination.

        Returns
        -------
        r: numpy array
            An array with the values of recombination.
        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)
        p = get_p(self.sys, self.efp, self.v, sites)

        ni2 = self.sys.ni[sites]**2
        n1 = self.sys.n1[sites]
        p1 = self.sys.p1[sites]
        tau_h = self.sys.tau_h[sites]
        tau_e = self.sys.tau_e[sites]
        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        r = (n*p - ni2)/(tau_h * (n+n1) + tau_e*(p+p1))
        return r

    def auger_rr(self, location=None):
        """
        Compute the Auger recombination across the system or on a line defined
        by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the recombination.

        Returns
        -------
        r: numpy array
            An array with the values of recombination.
        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)

        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        ni2 = self.sys.ni[sites]**2
        r = self.sys.Cn[sites] * n * (n*p - ni2) + self.sys.Cp[sites] * p * (n*p - ni2)
        return r

    def radiative_rr(self, location=None):
        """
        Compute the radiative recombination across the system or on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the recombination.

        Returns
        -------
        r: numpy array
            An array with the values of recombination.
        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)

        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        ni2 = self.sys.ni[sites]**2
        r = self.sys.B[sites] * (n*p - ni2)
        return r

    def electron_current(self, component='x', location=None):
        """
        Compute the electron current either by component (x or y) across the
        entire system, or on a line defined by two points.

        Parameters
        ----------
        component: string
            Current direction ``'x'`` or ``'y'``. By default returns all currents
            in the x-direction.
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the electron
            current.

        Returns
        -------
        jn: numpy array of floats
        """

        if location is not None:
            p1, p2 = location
            X, sites = self.line(self.sys, p1, p2)
            jn = get_jn(self.sys, self.efn, self.v, sites[:-1], sites[1:], X[1:]-X[:-1])
        else:
            Nx, Ny = self.sys.nx, self.sys.ny
            sites = self.sites.reshape(Ny, Nx)
            if component == 'x':
                sites = sites[:Ny, :Nx-1].flatten()
                dx = np.tile(self.sys.dx, Ny)
                jn = get_jn(self.sys, self.efn, self.v, sites, sites+1, dx)
            if component == 'y':
                sites = sites[:Ny-1, :Nx].flatten()
                dy = np.repeat(self.sys.dy, Nx)
                jn = get_jn(self.sys, self.efn, self.v, sites, sites+Nx, dy)
        return jn

    def hole_current(self, component='x', location=None):
        """
        Compute the hole current either by component (x or y) across the entire
        system, or on a line defined by two points.

        Parameters
        ----------
        component: string
            Current direction ``'x'`` or ``'y'``. By default returns all currents
            in the x-direction.
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the hole
            current.

        Returns
        -------
        jp: numpy array of floats
        """

        if location is not None:
            p1, p2 = location
            X, sites = self.line(self.sys, p1, p2)
            jp = get_jp(self.sys, self.efp, self.v, sites[:-1], sites[1:], X[1:]-X[:-1])
        else:
            Nx, Ny = self.sys.nx, self.sys.ny
            sites = self.sites.reshape(Ny, Nx)
            if component == 'x':
                sites = sites[:Ny, :Nx-1].flatten()
                dx = np.tile(self.sys.dx, Ny)
                jp = get_jp(self.sys, self.efp, self.v, sites, sites+1, dx)
            if component == 'y':
                sites = sites[:Ny-1, :Nx].flatten()
                dy = np.repeat(self.sys.dy, Nx)
                jp = get_jp(self.sys, self.efp, self.v, sites, sites+Nx, dy)
        return jp

    def electron_current_map(self, cmap='gnuplot', scale=1e6):
        """
        Compute a 2D map of the electron current.

        Parameters
        ----------
        cmap: Matplotlib color map
            Color map used for the plot.
        scale: float
            Scale to apply to the axes of the plot.

        """
        self.current_map(True, cmap, scale)

    def hole_current_map(self, cmap='gnuplot', scale=1e6):
        """
        Compute a 2D map of the hole current of a 2D system.

        Parameters
        ----------
        cmap: Matplotlib color map
            Color map used for the plot.
        scale: float
            Scale to apply to the axes of the plot.

        """
        self.current_map(False, cmap, scale)

    def current_map(self, electron, cmap, scale, fig=None):

        if not mpl_enabled:
            raise RuntimeError("matplotlib was not found, but is required "
                               "for plotting.")

        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        # add axis to figure
        ax = fig.add_subplot(111)

        Lx = self.sys.xpts[-2] * scale
        Ly = self.sys.ypts[-2] * scale

        x, y = self.sys.xpts[:-1], self.sys.ypts[:-1]
        nx, ny = len(x), len(y)
        
        s = np.asarray([i + j*self.sys.nx for j in range(self.sys.ny-1)\
                                     for i in range(self.sys.nx-1)])
        dx = np.tile(self.sys.dx, ny)
        dy = np.repeat(self.sys.dy, nx)

        if electron:
            Jx = get_jn(self.sys, self.efn, self.v, s, s+1, dx)
            Jy = get_jn(self.sys, self.efn, self.v, s, s+(nx+1), dy)
        else:
            Jx = get_jp(self.sys, self.efp, self.v, s, s+1, dx)
            Jy = get_jp(self.sys, self.efp, self.v, s, s+(nx+1), dy)

        Jx = np.reshape(Jx, (ny, nx))
        Jy = np.reshape(Jy, (ny, nx))

        jx = interp2d(x*scale, y*scale, Jx, kind='linear')
        jy = interp2d(x*scale, y*scale, Jy, kind='linear')

        xx, yy = np.linspace(0, Lx, 100), np.linspace(0, Ly, 100)
        jnx, jny = jx(xx, yy), jy(xx, yy)
        norm = np.sqrt(jnx**2 + jny**2)

        y, x = np.mgrid[0:Ly:100j, 0:Lx:100j]
        p = ax.pcolor(x, y, norm, cmap=cmap, rasterized=True)
        cbar = fig.colorbar(p, ax=ax)

        ax.streamplot(x, y, jnx, jny, linewidth=1, color='#a9a9a9', density=2)
        ax.set_xlim(xmax=Lx, xmin=0)
        ax.set_ylim(ymin=0, ymax=Ly)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if show:
            plt.show()

    def map3D(self, data, cmap='gnuplot', scale=1e-6):
        """
        Plot a 3D map of data across the entire system.

        Parameters
        ----------

        data: numpy array
            One-dimensional array of data with size equal to the size of the system.
        cmap: string
            Name of the colormap used by Matplolib.
        scale: float
            Relevant scaling to apply to the axes.
        """

        if not mpl_enabled:
            raise RuntimeError("matplotlib was not found, but is required "
                               "for map3D()")
        if not has3d:
            raise RuntimeError("Installed matplotlib does not support 3d plotting")

        xpts, ypts = self.sys.xpts / scale, self.sys.ypts / scale
        nx, ny = len(xpts), len(ypts)
        data_xy = data.reshape(ny, nx).T
        X, Y = np.meshgrid(xpts, ypts)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1, projection='3d')
        Z = data_xy.T
        ax.plot_surface(X, Y, Z,  cmap=cmap)
        ax.mouse_init(rotate_btn=1, zoom_btn=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


    def bulk_srh_recombination_current(self):
        """
        Compute the bulk Shockley-Read-Hall recombination current.

        Returns
        -------
        JR: float
            The integrated bulk recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.bulk_recombination_current('srh')

    def bulk_auger_recombination_current(self):
        """
        Compute the bulk Auger recombination current.

        Returns
        -------
        JR: float
            The integrated bulk recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.bulk_recombination_current('auger')

    def bulk_radiative_recombination_current(self):
        """
        Compute the bulk radiative recombination current.

        Returns
        -------
        JR: float
            The integrated bulk recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.bulk_recombination_current('radiative')

    def bulk_recombination_current(self, mec):
        x = self.sys.xpts / self.sys.scaling.length
        if self.sys.ny > 1:
            y = self.sys.ypts / self.sys.scaling.length
        u = []
        for j in range(self.sys.ny):
            # List of sites
            s = [i + j*self.sys.nx for i in range(self.sys.nx)]

            # Carrier densities
            n = get_n(self.sys, self.efn, self.v, s)
            p = get_p(self.sys, self.efp, self.v, s)

            # Recombination
            ni2 = self.sys.ni[s]**2

            if mec == 'srh':
                n1 = self.sys.n1[s]
                p1 = self.sys.p1[s]
                tau_h = self.sys.tau_h[s]
                tau_e = self.sys.tau_e[s]
                r = (n*p - ni2)/(tau_h * (n+n1) + tau_e*(p+p1))
            if mec == 'auger':
                r = self.sys.Cn[s] * n * (n*p - ni2)\
                  + self.sys.Cp[s] * p * (n*p - ni2)
            if mec == 'radiative':
                r = self.sys.B[s] * (n*p - ni2)

            sp = spline(x, r)
            u.append(sp.integral(x[0], x[-1]))
        if self.sys.ny == 1:
            JR = u[-1]
        if self.sys.ny > 1:
            sp = spline(y, u)
            JR = sp.integral(y[0], y[-1])
        return JR
     
    def full_current(self):
        """
        Compute the steady state current in 1D and 2D.

        Returns
        -------
        J: float
            The integrated full steady state current.

        Warnings
        --------
        Not implemented in 3D.
        """
        # Define the sites between which computing the currents
        sites_i = [self.sys.nx//2 + j*self.sys.nx for j in range(self.sys.ny)]
        sites_ip1 = [self.sys.nx//2+1 + j*self.sys.nx for j in range(self.sys.ny)]
        # And the corresponding lattice dimensions
        dl = self.sys.dx[self.sys.nx//2]

        # Compute the electron and hole currents
        jn = get_jn(self.sys, self.efn, self.v, sites_i, sites_ip1, dl)
        jp = get_jp(self.sys, self.efp, self.v, sites_i, sites_ip1, dl)

        if self.sys.ny == 1:
            j = jn + jp
        if self.sys.ny > 1:
            # Interpolate the results and integrate over the y-direction
            y = self.sys.ypts / self.sys.scaling.length
            j = spline(y, jn+jp).integral(y[0], y[-1])

        return j
