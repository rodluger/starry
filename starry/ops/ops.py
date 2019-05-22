from .. import _c_ops
from .integration import sT
from .rotation import dotRxy, dotRxyT, dotRz
from .filter import F
from .utils import RAxisAngle, VectorRAxisAngle, cross, to_tensor, to_array
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


__all__ = ["STARRY_ORTHOGRAPHIC_PROJECTION",
           "STARRY_RECTANGULAR_PROJECTION",
           "Ops"]


STARRY_ORTHOGRAPHIC_PROJECTION = 0
STARRY_RECTANGULAR_PROJECTION = 1


class autocompile(object):
    """
    A decorator used to automatically compile methods into Theano functions
    if the user disables lazy evaluation.

    """
    def __init__(self, name, *args):
        """
        Initialize the decorator.

        Arguments:
            name (str): The name of the decorated method
            *args (tuple): Theano types corresponding to each of the
                arguments of the method.
        """
        self.args = args
        self.name = name
        self.compiled_name = "_compiled_{0}".format(name)

    def __call__(self, func):
        """
        Wrap the method `func` and return a compiled version if `lazy==False`.
        
        """
        def wrapper(instance, *args, no_compile=False):
            """
            The magic happens in here.

            """
            if instance.lazy or no_compile:
                # Just return the function as is
                return func(instance, *args)
            else:
                # Compile the function if needed & cache it
                if not hasattr(instance, self.compiled_name):
                    logger.handlers[0].terminator = ""
                    logger.info("Compiling method `{0}`... ".format(self.name))
                    compiled_func = theano.function(
                        [*self.args], 
                        func(instance, *self.args), 
                        on_unused_input='ignore'
                    )
                    setattr(instance, self.compiled_name, compiled_func)
                    logger.handlers[0].terminator = "\n"
                    logger.info("Done.")
                # Return the compiled version
                return getattr(instance, self.compiled_name)(*args)
        return wrapper


class Ops(object):
    """
    Everything in radians here.

    """

    def __init__(self, ydeg, udeg, fdeg, lazy, quiet=False):
        """

        """
        # Logging
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Instantiate the C++ Ops
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self.filter = (fdeg > 0) or (udeg > 0)
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)
        self.lazy = lazy
        if self.lazy:
            self.cast = to_tensor
        else:
            self.cast = to_array

        # Solution vectors
        self.sT = sT(self._c_ops.sT, self._c_ops.N)
        self.rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT))
        self.rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1))
        
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

        # mu, nu arrays
        deg = self.ydeg + self.udeg + self.fdeg
        N = (deg + 1) ** 2
        self._mu = np.zeros(N, dtype=int)
        self._nu = np.zeros(N, dtype=int)
        n = 0
        for l in range(deg + 1):
            for m in range(-l, l + 1):
                self._mu[n] = l - m
                self._nu[n] = l + m
                n += 1
        self._mu = tt.as_tensor_variable(self._mu)
        self._nu = tt.as_tensor_variable(self._nu)

    def compute_ortho_pT(self, res):
        """
        Compute the polynomial basis on the plane of the sky.

        """
        dx = 2.0 / res
        y, x = tt.mgrid[-1:1:dx, -1:1:dx]
        x = tt.reshape(x, [-1])
        y = tt.reshape(y, [-1])
        z = tt.sqrt(1 - x ** 2 - y ** 2)
        return self.pT(x, y, z)

    def compute_rect_pT(self, res):
        """
        Compute the polynomial basis on a rectangular lat/lon grid.

        """
        dx = np.pi / res
        lat, lon = tt.mgrid[-np.pi/2:np.pi/2:dx, -3*np.pi/2:np.pi/2:2*dx]
        x = tt.reshape(tt.cos(lat) * tt.cos(lon), [1, -1])
        y = tt.reshape(tt.cos(lat) * tt.sin(lon), [1, -1])
        z = tt.reshape(tt.sin(lat), [1, -1])
        R = RAxisAngle([1, 0, 0], -np.pi / 2)
        xyz = tt.dot(R, tt.concatenate((x, y, z)))
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        return self.pT(x, y, z)

    def pT(self, x, y, z):
        """
        TODO: Can probably be sped up with recursion.

        """
        def _pT_step(mu, nu, x, y, z):
            return tt.switch(
                tt.eq((nu % 2), 0), 
                x ** (mu / 2) * y ** (nu / 2), 
                x ** ((mu - 1) / 2) * y ** ((nu - 1) / 2) * z
            )
        pT, updates = theano.scan(fn=_pT_step,
                                  sequences=[self._mu, self._nu],
                                  non_sequences=[x, y, z]
        )
        return tt.transpose(pT)

    def dotR(self, M, inc, obl, theta):
        """

        """

        res = self.dotRxyT(M, inc, obl)
        res = self.dotRz(res, theta)
        res = self.dotRxy(res, inc, obl)
        return res

    @autocompile(
        "X",
        tt.dvector("theta"), 
        tt.dvector("xo"), 
        tt.dvector("yo"), 
        tt.dvector("zo"), 
        tt.dscalar("ro"), 
        tt.dscalar("inc"),
        tt.dscalar("obl"),
        tt.dvector("u"), 
        tt.dvector("f")
    )
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

    @autocompile(
        "flux",
        tt.dvector("theta"), 
        tt.dvector("xo"), 
        tt.dvector("yo"), 
        tt.dvector("zo"), 
        tt.dscalar("ro"), 
        tt.dscalar("inc"),
        tt.dscalar("obl"),
        tt.dvector("y"), 
        tt.dvector("u"), 
        tt.dvector("f")
    )
    def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f):
        """

        """
        return tt.dot(self.X(theta, xo, yo, zo, ro, inc, obl, u, f, no_compile=True), y)

    @autocompile(
        "intensity",
        tt.dvector("xpt"), 
        tt.dvector("ypt"), 
        tt.dvector("zpt"), 
        tt.dvector("y"), 
        tt.dvector("u"), 
        tt.dvector("f")
    )
    def intensity(self, xpt, ypt, zpt, y, u, f):
        """

        """
        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            A1y = ts.dot(self.F(u, f), A1y)

        # Dot the polynomial into the basis
        return tt.dot(pT, A1y)
    
    @autocompile(
        "render",
        tt.iscalar("res"), 
        tt.iscalar("projection"), 
        tt.dvector("theta"), 
        tt.dscalar("inc"), 
        tt.dscalar("obl"), 
        tt.dvector("y"), 
        tt.dvector("u"), 
        tt.dvector("f")
    )
    def render(self, res, projection, theta, inc, obl, y, u, f):
        """

        """
        # Compute the polynomial basis
        pT = tt.switch(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_pT(res),
            self.compute_ortho_pT(res)
        )

        # If lat/lon, rotate the map so that north points up
        y = tt.switch(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            tt.reshape(
                self.dotRxy(
                    self.dotRz(
                        tt.reshape(y, [1, -1]), tt.reshape(-obl, [1])
                    ), np.pi / 2 - inc, 0
                ), [-1]
            ),
            y
        )

        # Rotate the map and transform into the polynomial basis
        yT = tt.tile(y, [theta.shape[0], 1])
        Ry = tt.transpose(self.dotR(yT, inc, obl, -theta))
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter
        if self.filter:
            A1Ry = tt.switch(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                ts.dot(self.F(u, f), A1Ry),
                A1Ry
            )

        # Dot the polynomial into the basis
        return tt.reshape(tt.dot(pT, A1Ry), [res, -1, theta.shape[0]])
    
    @autocompile(
        "get_inc_obl",
        tt.dvector("axis")
    )
    def get_inc_obl(self, axis):
        """

        """
        axis /= axis.norm(2)
        inc_obl = tt.zeros(2)
        obl = tt.arctan2(axis[0], axis[1])
        sino = tt.sin(obl)
        coso = tt.cos(obl)
        inc = tt.switch(
            tt.lt(tt.abs_(sino), 1e-10),
            tt.arctan2(axis[1] / coso, axis[2]),
            tt.arctan2(axis[0] / sino, axis[2])
        )
        inc_obl = tt.set_subtensor(inc_obl[0], inc)
        inc_obl = tt.set_subtensor(inc_obl[1], obl)
        return inc_obl
    
    @autocompile(
        "get_axis",
        tt.dscalar("inc"),
        tt.dscalar("obl")
    )
    def get_axis(self, inc, obl):
        """

        """
        axis = tt.zeros(3)
        sino = tt.sin(obl)
        coso = tt.cos(obl)
        sini = tt.sin(inc)
        cosi = tt.cos(inc)
        axis = tt.set_subtensor(axis[0], sino * sini)
        axis = tt.set_subtensor(axis[1], coso * sini)
        axis = tt.set_subtensor(axis[2], cosi)
        return axis
    
    def set_map_vector(self, vector, inds, vals):
        """

        """
        return tt.set_subtensor(vector[inds], vals * tt.ones(len(inds)))

    @autocompile(
        "latlon_to_xyz",
        tt.dvector("axis"),
        tt.dvector("lat"),
        tt.dvector("lon")
    )
    def latlon_to_xyz(self, axis, lat, lon):
        """

        """
        # Get the `lat = 0, lon = 0` point
        u = [axis[1], -axis[0], 0]
        theta = tt.arccos(axis[2])
        R0 = RAxisAngle(u, theta)
        origin = tt.dot(R0, [0.0, 0.0, 1.0])

        # Now rotate it to `lat, lon`
        R1 = VectorRAxisAngle([1.0, 0.0, 0.0], -lat)
        R2 = VectorRAxisAngle([0.0, 1.0, 0.0], lon)
        R = tt.batched_dot(R2, R1)
        xyz = tt.transpose(tt.dot(R, origin))
        return xyz
    
    @autocompile(
        "rotate",
        tt.dvector("y"),
        tt.dvector("theta"),
        tt.dscalar("inc"),
        tt.dscalar("obl")
    )
    def rotate(self, y, theta, inc, obl):
        """

        """
        return tt.reshape(self.dotR(y.reshape([1, -1]), inc, obl, -theta), [-1])
    
    @autocompile(
        "align",
        tt.dvector("y"),
        tt.dvector("source"),
        tt.dvector("dest")
    )
    def align(self, y, source, dest):
        """

        """
        source /= source.norm(2)
        dest /= dest.norm(2)
        axis = cross(source, dest)
        inc_obl = self.get_inc_obl(axis, no_compile=True)
        inc = inc_obl[0]
        obl = inc_obl[1]
        theta = tt.reshape(tt.arccos(tt.dot(source, dest)), [1])
        return ifelse(
            tt.all(tt.eq(source, dest)),
            y,
            self.rotate(y, theta, inc, obl, no_compile=True)
        )