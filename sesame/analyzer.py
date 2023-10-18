# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from scipy.interpolate import InterpolatedUnivariateSpline as spline, interp2d
from .utils import Bresenham, get_indices
from .observables import *
from .defects import defectsF

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
        self.sites = np.arange(sys.nx*sys.ny, dtype=int)

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
        s, x, _, _ = Bresenham(system, p1, p2)
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
            idx1, _ = get_indices(self.sys, (p1[0],0,0))
            idx2, _ = get_indices(self.sys, (p2[0],0,0))
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
        X = X * 1e4  # in um

        l1, = ax.plot(X, vt*self.efn[sites], lw=2, color='#2e89cf', ls='--')
        l2, = ax.plot(X, vt*self.efp[sites], lw=2, color='#cf392e', ls='--')
        l3, = ax.plot(X, -vt * (self.v[sites] + self.sys.bl[sites]), lw=2, color='k', ls='-')
        l4, = ax.plot(X, -vt * (self.v[sites] + self.sys.bl[sites] + self.sys.Eg[sites]), lw=2, color='k', ls='-')

        fig.legend([l1, l2], [r'$\mathregular{E_{F_n}}$',\
                              r'$\mathregular{E_{F_p}}$'])


        ax.set_xlabel(r'Position [$\mathregular{\mu m}$]')
        ax.set_ylabel('Energy [eV]')

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

    def defect_rr(self, defect):
        """
        Compute the recombination for all sites of a defect (2D and 3D).

        Parameters
        ----------
        defect: named tuple
            Container with the properties of a defect. The expected field names
            of the named tuple are sites, location, dos, energy, sigma_e,
            sigma_h, transition, perp_dl.

        Returns
        -------
        r: numpy array of floats
            An array with the values of recombination at each sites.
        """

        # Create arrays to pass to defectsF
        n = self.electron_density()
        p = self.hole_density()
        rho = np.zeros_like(n) 
        r = np.zeros_like(n)

        # Update r (and rho but we don't use it)
        defectsF(self.sys, [defect], n, p, rho, r=r)
        r = np.multiply(r[defect.sites],defect.perp_dl)

        return r

    def total_rr(self):
        """
        Compute the sum of all the recombination sources for all sites of the
        system.

        Returns
        -------
        r: numpy array of floats
            An array with the values of the total recombination at each sites.
        """

        srh = self.bulk_srh_rr()
        radiative = self.radiative_rr()
        auger = self.auger_rr()
        defects = np.zeros_like(srh)
        for defect in self.sys.defects_list:
            sites = defect.sites
            defects[sites] += self.defect_rr(defect)

        return srh + radiative + auger + defects


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

    def electron_current_map(self, cmap='gnuplot', scale=1e4):
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

    def hole_current_map(self, cmap='gnuplot', scale=1e4):
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
        dy = np.repeat(self.sys.dy[:-1], nx)

        if electron:
            Jx = get_jn(self.sys, self.efn, self.v, s, s+1, dx)
            Jy = get_jn(self.sys, self.efn, self.v, s, s+(nx+1), dy)
            title = r'$\mathregular{J_{n}\ [mA\cdot cm^{-2}]}$'
        else:
            Jx = get_jp(self.sys, self.efp, self.v, s, s+1, dx)
            Jy = get_jp(self.sys, self.efp, self.v, s, s+(nx+1), dy)
            title = r'$\mathregular{J_{p}\ [mA\cdot cm^{-2}]}$'

        Jx = np.reshape(Jx, (ny, nx)) * self.sys.scaling.current * 1e3
        Jy = np.reshape(Jy, (ny, nx)) * self.sys.scaling.current * 1e3

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

        ax.set_xlabel(r'x [$\mathregular{\mu m}$]')
        ax.set_ylabel(r'y [$\mathregular{\mu m}$]')
        ax.set_title(title)

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


    def integrated_bulk_srh_recombination(self):
        """
        Integrate the bulk Shockley-Read-Hall recombination over an entire
        system.

        Returns
        -------
        JR: float
            The integrated bulk recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.integrated_recombination('srh')

    def integrated_auger_recombination(self):
        """
        Integrate the Auger recombination over an entire system.

        Returns
        -------
        JR: float
            The integrated Auger recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.integrated_recombination('auger')

    def integrated_radiative_recombination(self):
        """
        Integrate the radiative recombination over an entire system.

        Returns
        -------
        JR: float
            The integrated radiative recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.integrated_recombination('radiative')

    def integrated_recombination(self, mec):
        # Compute recombination averywhere
        if mec == 'srh':
            r = self.bulk_srh_rr()
        if mec == 'auger':
            r = self.auger_rr()
        if mec == 'radiative':
            r = self.radiative_rr()

        # Integrate along x for each y (if any
        x = self.sys.xpts / self.sys.scaling.length
        if self.sys.ny > 1:
            y = self.sys.ypts / self.sys.scaling.length
        u = []
        for j in range(self.sys.ny):
            # List of sites
            s = [i + j*self.sys.nx for i in range(self.sys.nx)]
            sp = spline(x, r[s])
            u.append(sp.integral(x[0], x[-1]))
        if self.sys.ny == 1:
            JR = u[-1]
        if self.sys.ny > 1:
            sp = spline(y, u)
            JR = sp.integral(y[0], y[-1])
        return JR
     
    def integrated_defect_recombination(self, defect):
        """
        Integrate the recombination along a defect in 2D.

        Returns
        -------
        JD: float
            The recombination integrated along the line of the defect.

        Warnings
        --------
        Not implemented in 3D.
        """
        # Find the path along which to integrate
        p1 = defect.location[0]
        p2 = defect.location[1]
        X, _ = self.line(self.sys, p1, p2)

        # interpolate recombination and integrate
        r = self.defect_rr(defect)
        sp = spline(X, r)
        JD = sp.integral(X[0], X[-1])

        return JD


    def full_current(self):
        """
        Compute the steady state current in 1D and 2D.

        Returns
        -------
        J: float
            The integrated full steady state current.
        """

        # System number of sites
        nx, ny= self.sys.nx, self.sys.ny

        # Define the sites between which computing the currents
        sites_i = [nx//2 + j*nx  for j in range(ny)]
        sites_ip1 = [nx//2+1+j*nx  for j in range(ny)]
        # And the corresponding lattice dimensions
        dl = self.sys.dx[self.sys.nx//2]

        # Compute the electron and hole currents
        jn = get_jn(self.sys, self.efn, self.v, sites_i, sites_ip1, dl)
        jp = get_jp(self.sys, self.efp, self.v, sites_i, sites_ip1, dl)

        if ny == 1:
            j = jn[0] + jp[0]
        if ny > 1:
            # Interpolate the results and integrate over the y-direction
            y = self.sys.ypts / self.sys.scaling.length
            j = spline(y, jn+jp).integral(y[0], y[-1])


        return j
