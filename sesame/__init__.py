
from ._version import __version__

__all__ = ['builder']
for module in __all__:
    exec('from . import {0}'.format(module))

available = [('builder', ['Builder']),
             ('solvers', ['ddp_solver', 'poisson_solver'])]
for module, names in available:
    exec('from .{0} import {1}'.format(module, ', '.join(names)))
    __all__.extend(names)

try:
    from . import plotter
    from .plotter import plot_line_defects, map2D, plot_plane_defects
except:
    pass
else:
    __all__.extend(['plotter', 'plot_line_defects', 'map2D',
    'plot_plane_defects'])

