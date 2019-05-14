from .. import _c_ops
from .sT import sT
from .dotRz import dotRz
import theano.tensor as tt
import theano.sparse as ts


class Ops(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, fdeg=0):
        # Instantiate the C++ Ops
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)

        # Solution vector
        self.sT = sT(self._c_ops)

        # Change of basis
        self.A = ts.as_sparse_variable(self._c_ops.A())

        # Rz rotation
        self.dotRz = dotRz(self._c_ops)