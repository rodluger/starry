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
        self.inc = 90.0
        self.obl = 0.0

    def flux(self, theta, xo, yo, ro):

        # Compute impact parameter and occultor angular position
        b = tt.sqrt(xo ** 2 + yo ** 2)
        theta_z = tt.arctan2(xo, yo) * (180.0 / np.pi)

        # Instantiate the ops
        sT = self._ops.sT(b, ro)
        A = self._ops.A
        rTA1 = self._ops.rTA1
        dotRz = self._ops.dotRz
        dotRxy = self._ops.dotRxy
        dotRxyT = self._ops.dotRxyT

        # Rotation only
        rTA1R = dotRxyT(rTA1, self.inc, self.obl)
        rTA1R = dotRz(rTA1R, theta)
        rTA1R = dotRxy(rTA1R, self.inc, self.obl)

        # Occultation + rotation
        sTA = ts.dot(sT, A)
        sTAR = dotRz(sTA, theta_z)
        sTARR = dotRxyT(sTAR, self.inc, self.obl)
        sTARR = dotRz(sTARR, theta)
        sTARR = dotRxy(sTARR, self.inc, self.obl)

        # Compute the design matrix
        X = tt.switch(
            (tt.ge(b, 1 + ro) | tt.eq(ro, 0.0))[:, None],
            rTA1R,
            sTARR
        )

        return tt.dot(X, self.y)
