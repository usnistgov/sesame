#!/usr/bin/env python

# __all__ = ['observables', 'nrsolver']

__all__ = ['builder', 'observables', 'solvers', 'nrsolver']
for module in __all__:
    exec('from . import {0}'.format(module))

available = [('builder', ['Builder']),
             ('solvers', ['ddp_solver', 'poisson_solver'])]
for module, names in available:
    exec('from .{0} import {1}'.format(module, ', '.join(names)))
    __all__.extend(names)
