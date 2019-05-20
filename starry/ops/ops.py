from .. import _c_ops
from .integration import sT
from .sht import pT
from .rotation import dotRxy, dotRxyT, dotRz
from .filter import F
from ..Map.utils import RAxisAngle
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
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self.filter = (fdeg > 0) or (udeg > 0)
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)

        # Solution vectors
        self.sT = sT(self._c_ops.sT, self._c_ops.N)
        self.rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT))
        self.rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1))
        self.pT = pT(self._c_ops.pT, self._c_ops.N)

        # Change of basis matrices
        self.A = ts.as_sparse_variable(self._c_ops.A)
        self.A1 = ts.as_sparse_variable(self._c_ops.A1)
        self.A1Inv = ts.as_sparse_variable(self._c_ops.A1Inv)

        # Rotation left-multiply operations
        self.dotRz = dotRz(self._c_ops.dotRz)
        self.dotRxy = dotRxy(self._c_ops.dotRxy)
        self.dotRxyT = dotRxyT(self._c_ops.dotRxyT)

        # Filter
        self.F = F(self._c_ops.F, self._c_ops.N, self._c_ops.Ny)

        # Map rendering
        self.rect_res = 0
        self.ortho_res = 0

    def dotR(self, M, inc, obl, theta):
        """

        """

        res = self.dotRxyT(M, inc, obl)
        res = self.dotRz(res, theta)
        res = self.dotRxy(res, inc, obl)
        return res


    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f):
        """

        """
        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = (tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0))
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]

        # Determine shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]

        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Rotation operator
        if self.filter:
            rTA1 = ts.dot(tt.dot(self.rT, F), self.A1)
        else:
            rTA1 = self.rTA1
        X_rot = tt.zeros((rows, cols))
        X_rot = tt.set_subtensor(
            X_rot[i_rot], 
            self.dotR(rTA1, inc, obl, theta[i_rot])
        )

        # Occultation + rotation operator
        X_occ = tt.zeros((rows, cols))
        sT = self.sT(b[i_occ], ro)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ])
        sTAR = self.dotRz(sTA, theta_z)
        if self.filter:
            A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
            sTAR = tt.dot(sTAR, A1InvFA1)
        X_occ = tt.set_subtensor(
            X_occ[i_occ], 
            self.dotR(sTAR, inc, obl, theta[i_occ])
        )

        return X_rot + X_occ


    def intensity(self, theta, xpt, ypt, zpt, inc, obl, y, u, f):
        """

        """
        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Compute the polynomial basis
        pT = self.pT(xpt, ypt, zpt)

        # Rotate the map and transform to polynomial
        yT = tt.tile(y, [theta.size, 1])
        Ry = tt.transpose(self.dotR(yT, inc, obl, -theta))
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter
        if self.filter:
            A1Ry = ts.dot(F, A1Ry)

        # Dot the polynomial into the basis
        return tt.reshape(tt.dot(pT, A1Ry), [xpt.shape[0], theta.shape[0]])
    

    def render(self, res, projection, theta, inc, obl, y, u, f):
        """

        """
        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Compute the polynomial basis
        if (projection == "rect") and (res != self.rect_res):
                self.rect_res = res
                lon = np.linspace(-np.pi, np.pi, 2 * res) - np.pi / 2
                lat = np.linspace(-np.pi / 2, np.pi / 2, res)
                lon, lat = np.meshgrid(lon, lat)
                xg = (np.cos(lat) * np.cos(lon)).reshape(1, -1)
                yg = (np.cos(lat) * np.sin(lon)).reshape(1, -1)
                zg = np.sin(lat).reshape(1, -1)
                R = RAxisAngle([1, 0, 0], -90)
                xg, yg, zg = np.dot(R, np.vstack((xg, yg, zg)))
                self.rect_pT = self.pT(xg.flatten(), yg.flatten(), zg.flatten()).eval()
        elif (projection == "ortho") and (res != self.ortho_res):
                self.ortho_res = res
                arr = np.linspace(-1, 1, self.ortho_res)
                xg, yg = np.meshgrid(arr, arr)
                zg = np.sqrt(1 - xg ** 2 - yg ** 2)
                self.ortho_pT = self.pT(xg.flatten(), yg.flatten(), zg.flatten()).eval()

        # Rotate the map and transform to polynomial
        yT = tt.tile(y, [theta.shape[0], 1])
        Ry = tt.transpose(self.dotR(yT, inc, obl, -theta))
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter
        if self.filter:
            A1Ry = ts.dot(F, A1Ry)

        # Dot the polynomial into the basis
        if projection == "rect":
            return tt.reshape(tt.dot(self.rect_pT, A1Ry), [res, 2 * res, theta.shape[0]])
        else:
            return tt.reshape(tt.dot(self.ortho_pT, A1Ry), [res, res, theta.shape[0]])
    