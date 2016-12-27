
from ._version import __version__

__all__ = ['builder', 'analyzer']
for module in __all__:
    exec('from . import {0}'.format(module))

available = [('builder', ['Builder']),
             ('solvers', ['solve', 'IVcurve']),
             ('analyzer', ['Analyzer'])]
for module, names in available:
    exec('from .{0} import {1}'.format(module, ', '.join(names)))
    __all__.extend(names)

try:
    from . import plotter
    from .plotter import plot_line_defects, plot, plot_plane_defects
except:
    pass
else:
    __all__.extend(['plotter', 'plot_line_defects', 'plot',
    'plot_plane_defects'])

