from .. import _c_ops
from .integration import sT
from .rotation import dotRxy, dotRxyT, dotRz
import theano.tensor as tt
import theano.sparse as ts
import numpy as np


class Ops(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, fdeg=0):
        """

        """

        # Instantiate the C++ Ops
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)

        # Solution vectors
        self.sT = sT(self._c_ops)
        self.rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT()))
        self.rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1()))

        # Change of basis matrices
        self.A = ts.as_sparse_variable(self._c_ops.A())
        self.A1 = ts.as_sparse_variable(self._c_ops.A1())

        # Rotation left-multiply operations
        self.dotRz = dotRz(self._c_ops)
        self.dotRxy = dotRxy(self._c_ops)
        self.dotRxyT = dotRxyT(self._c_ops)

    
    def dotR(self, M, inc, obl, theta):
        """

        """

        res = self.dotRxyT(M, inc, obl)
        res = self.dotRz(res, theta)
        res = self.dotRxy(res, inc, obl)
        return res


    def X(self, theta, xo, yo, ro, inc, obl):
        """

        """

        # TODO: Handle shapes & types, etc.
        theta = tt.as_tensor_variable(theta)
        xo = tt.as_tensor_variable(xo)
        yo = tt.as_tensor_variable(yo)

        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = (tt.ge(b, 1 + ro) | tt.eq(ro, 0.0))
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]

        # Shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]

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