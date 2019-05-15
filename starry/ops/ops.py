from .. import _c_ops
from .sT import sT
from .dotRz import dotRz
from .dotRxy import dotRxy
import theano.tensor as tt
import theano.sparse as ts


class Ops(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, fdeg=0):

        # Instantiate the C++ Ops
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)

        # Solution vectors
        self.sT = sT(self._c_ops)
        self.rT = tt.as_tensor_variable(self._c_ops.rT())
        self.rTA1 = tt.as_tensor_variable(self._c_ops.rTA1())

        # Change of basis
        self.A = ts.as_sparse_variable(self._c_ops.A())
        self.A1 = ts.as_sparse_variable(self._c_ops.A1())

        # Rotations
        self.dotRz = dotRz(self._c_ops)
        self.dotRxy = dotRxy(self._c_ops)