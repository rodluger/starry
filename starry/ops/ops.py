from .. import _c_ops
from .integration import sT
from .rotation import dotRxy, dotRxyT, dotRz
from .filter import F
import theano.tensor as tt
import theano.sparse as ts
import numpy as np


class Ops(object):
    """

    """

    def __init__(self, ydeg, udeg, fdeg):
        """

        """

        # Instantiate the C++ Ops
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)

        # Solution vectors
        self.sT = sT(self._c_ops.sT, self._c_ops.N)
        self.rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT))
        self.rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1))

        # Change of basis matrices
        self.A = ts.as_sparse_variable(self._c_ops.A)
        self.A1 = ts.as_sparse_variable(self._c_ops.A1)

        # Rotation left-multiply operations
        self.dotRz = dotRz(self._c_ops.dotRz)
        self.dotRxy = dotRxy(self._c_ops.dotRxy)
        self.dotRxyT = dotRxyT(self._c_ops.dotRxyT)

        # Filter
        self.F = F(self._c_ops.F, self._c_ops.N, self._c_ops.Ny)

    
    def dotR(self, M, inc, obl, theta):
        """

        """

        res = self.dotRxyT(M, inc, obl)
        res = self.dotRz(res, theta)
        res = self.dotRxy(res, inc, obl)
        return res


    def X(self, theta, xo, yo, zo, ro, inc, obl, u):
        """

        """

        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = (tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0))
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]

        # Shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]

        # TODO: Filter (u, f)

        # Rotation operator
        X_rot = tt.zeros((rows, cols))
        X_rot = tt.set_subtensor(
            X_rot[i_rot], 
            self.dotR(self.rTA1, inc, obl, theta[i_rot])
        )

        # Occultation + rotation operator
        X_occ = tt.zeros((rows, cols))
        sT = self.sT(b[i_occ], ro)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ]) * (180.0 / np.pi)
        sTAR = self.dotRz(sTA, theta_z)
        X_occ = tt.set_subtensor(
            X_occ[i_occ], 
            self.dotR(sTAR, inc, obl, theta[i_occ])
        )

        return X_rot + X_occ