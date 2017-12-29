# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

# ----------------------------------------------------------------------------- 
# This file is a modified version of PyMUMPS written by Bradley Froehle. The
# licence and conditions of use of PyMUMPS are reproduced below.
# ----------------------------------------------------------------------------- 
# Copyright (c) 2013, Bradley Froehle <brad.froehle@gmail.com>
# All rights reserved.
#
# This file is part of PyMUMPS (https://pypi.python.org/pypi/PyMUMPS), and
# subject to the following terms of use:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
from . import _dmumps

__all__ = [
    'DMumpsContext',
    'spsolve',
    ]

########################################################################
# Classes
########################################################################

# The main class which is shared between the various datatype variants.
class _MumpsBaseContext(object):
    """MUMPS Context

    This context acts as a thin wrapper around MUMPS_STRUC_C
    which is accessible in the `id` attribute.

    Usage
    -----

    Basic usage generally involves setting up the context, adding
    the sparse matrix and right hand side in process 0, and using
    `run` to execute the various MUMPS phases.

        ctx = MumpsContext()
        ctx.set_sparse(A)
        x = b.copy() # MUMPS modifies rhs in place, so make copy
        ctx.set_rhs(x)
        ctx.run(6) # Symbolic + Numeric + Solve
        ctx.destroy() # Free internal data structures

        assert abs(A.dot(x) - b).max() < 1e-10
    """

    def __init__(self, sym=0):
        """Create a MUMPS solver context.

        Parameters
        ----------
        sym : int
            0 if unsymmetric
        """
        self.id = self._MUMPS_STRUC_C()
        self.id.par = 1
        self.id.sym = sym
        self.id.comm_fortran = -987654
        self.run(job = -1) # JOB_INIT
        self._refs = {} # References to matrices

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.destroy()

    def set_shape(self, n):
        """Set the matrix shape."""
        self.id.n = n

    def set_sparse(self, A):
        """Set assembled matrix on processor 0.

        Parameters
        ----------
        A : `scipy.sparse.coo_matrix`
            Sparse matrices of other formats will be converted to
            COOrdinate form.
        """
        # if self.myid != 0:
        #     return

        A = A.tocoo()
        n = A.shape[0]
        assert A.shape == (n, n), "Expected a square matrix."
        self.set_shape(n)
        self.set_assembled(A.row+1, A.col+1, A.data)


    ####################################################################
    # Supplies the matrix
    ####################################################################

    def set_assembled(self, irn, jcn, a):
        """Set assembled matrix.

        The row and column indices (irn & jcn) should be one based.
        """
        self.set_assembled_rows_cols(irn, jcn)
        self.set_assembled_values(a)

    def set_assembled_rows_cols(self, irn, jcn):
        """Set assembled matrix indices.

        The row and column indices (irn & jcn) should be one based.
        """
        assert irn.size == jcn.size
        self._refs.update(irn=irn, jcn=jcn)
        self.id.nz = irn.size
        self.id.irn = self.cast_array(irn)
        self.id.jcn = self.cast_array(jcn)

    def set_assembled_values(self, a):
        """Set assembled matrix values."""
        assert a.size == self.id.nz
        self._refs.update(a=a)
        self.id.a = self.cast_array(a)

    ####################################################################
    # Right hand side entry
    ####################################################################

    def set_rhs(self, rhs):
        """Set the right hand side. This matrix will be modified in place."""
        assert rhs.size == self.id.n
        self._refs.update(rhs=rhs)
        self.id.rhs = self.cast_array(rhs)

    def set_icntl(self, idx, val):
        """Set the icntl value.

        The index should be provided as a 1-based number.
        """
        self.id.icntl[idx-1] = val

    def set_job(self, job):
        """Set the job."""
        self.id.job = job

    def set_silent(self):
        """Silence most messages."""
        self.set_icntl(1, -1) # output stream for error msgs
        self.set_icntl(2, -1) # otuput stream for diagnostic msgs
        self.set_icntl(3, -1) # output stream for global info
        self.set_icntl(4, 0)  # level of printing for errors

    @property
    def destroyed(self):
        return self.id is None

    def destroy(self):
        """Delete the MUMPS context and release all array references."""
        if self.id is not None and self._mumps_c is not None:
            self.id.job = -2 # JOB_END
            self._mumps_c(self.id)
        self.id = None
        self._refs = None

    def __del__(self):
        if not self.destroyed:
            warnings.warn("undestroyed %s" % self.__class__.__name__,
                          RuntimeWarning)
        self.destroy()

    def mumps(self):
        """Call MUMPS, checking for errors in the return code.

        The desired job should have already been set using `ctx.set_job(...)`.
        As a convenience, you may wish to call `ctx.run(job=...)` which sets
        the job and calls MUMPS.
        """
        self._mumps_c(self.id)
        if self.id.infog[0] < 0:
            raise RuntimeError("MUMPS error: %d" % self.id.infog[0])

    def run(self, job):
        """Set the job and run MUMPS.

        Valid Jobs
        ----------
        1 : Analysis
        2 : Factorization
        3 : Solve
        4 : Analysis + Factorization
        5 : Factorization + Solve
        6 : Analysis + Factorization + Solve
        """
        self.set_job(job)
        self.mumps()

class DMumpsContext(_MumpsBaseContext):

    cast_array = staticmethod(_dmumps.cast_array)
    _mumps_c = staticmethod(_dmumps.dmumps_c)
    _MUMPS_STRUC_C = staticmethod(_dmumps.DMUMPS_STRUC_C)


########################################################################
# Functions
########################################################################

def spsolve(A, b):
    """Sparse solve A\b."""
    assert A.dtype == 'd' and b.dtype == 'd', "Only double precision supported."
    with DMumpsContext(sym=0) as ctx:
        # Set the sparse matrix
        ctx.set_sparse(A.tocoo())
        x = b.copy()
        ctx.set_rhs(x)

        # Silence most messages
        ctx.set_silent()

        # Ordering package
        # 3: SCOTCH
        # 4: PORD
        # 5: METIS
        ctx.set_icntl(7, 4)

        # Analysis + Factorization + Solve
        ctx.run(job=6)

        return x
