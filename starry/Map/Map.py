from ..ops import Ops
import numpy as np
import theano
import theano.tensor as tt
import theano.sparse as ts


class Map(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, fdeg=0):
        """

        """

        # Instantiate the Theano ops class
        self.ops = Ops(ydeg=ydeg, udeg=udeg, fdeg=fdeg)

        self.y = np.zeros((ydeg + 1) ** 2)
        self.y[0] = 1.0
        self.inc = 90.0
        self.obl = 0.0

    def X(self, theta, xo, yo, ro):
        """
        Compute and return the light curve design matrix.

        """

        return self.ops.X(theta, xo, yo, ro, self.inc, self.obl)

    def flux(self, *args, **kwargs):
        """
        Compute and return the light curve.
        
        """

        return tt.dot(self.X(*args, **kwargs), self.y)