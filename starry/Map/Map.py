from ..ops import Ops
import numpy as np
import theano
import theano.tensor as tt
import theano.sparse as ts


class Map(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, fdeg=0):
        self._ops = Ops(ydeg=ydeg, udeg=udeg, fdeg=fdeg)

        self.y = np.zeros((ydeg + 1) ** 2)
        self.y[0] = 1.0

    def flux(self, xo, yo, ro):
        b = tt.sqrt(xo ** 2 + yo ** 2)
        sT = self._ops.sT(b, ro)
        A = self._ops.A

        sTA = ts.dot(sT, A)

        # TODO Rz = ?

        sTARz, _ = theano.scan(
            fn=lambda sTA, Rz: tt.dot(sTA, Rz), 
            sequences=[sTA, Rz]
        )

        return tt.dot(sTARz, self.y)

