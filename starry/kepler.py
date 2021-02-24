# -*- coding: utf-8 -*-
from . import config
from ._constants import *
from .maps import MapBase, RVBase, ReflectedBase
from ._core import OpsSystem, math
from .compat import evaluator
import numpy as np
from astropy import units
from inspect import getmro
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os
import logging

logger = logging.getLogger("starry.maps")


__all__ = ["Primary", "Secondary", "System"]


class Body(object):
    """A generic body. Must be subclassed."""

    def __init__(
        self,
        map,
        r=1.0,
        m=1.0,
        prot=1.0,
        t0=0.0,
        theta0=0.0,
        length_unit=units.Rsun,
        mass_unit=units.Msun,
        time_unit=units.day,
        angle_unit=units.degree,
        **kwargs,
    ):
        # Surface map
        self._lazy = map._lazy
        self._math = map._math
        self.map = map

        # Units
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.time_unit = time_unit
        self.angle_unit = angle_unit

        # Attributes
        self.r = r
        self.m = m
        self.prot = prot
        self.t0 = t0
        self.theta0 = theta0

    @property
    def length_unit(self):
        """An ``astropy.units`` unit defining the length metric for this body."""
        return self._length_unit

    @length_unit.setter
    def length_unit(self, value):
        assert value.physical_type == "length"
        self._length_unit = value
        self._length_factor = value.in_units(units.Rsun)

    @property
    def mass_unit(self):
        """An ``astropy.units`` unit defining the mass metric for this body."""
        return self._mass_unit

    @mass_unit.setter
    def mass_unit(self, value):
        assert value.physical_type == "mass"
        self._mass_unit = value
        self._mass_factor = value.in_units(units.Msun)

    @property
    def time_unit(self):
        """An ``astropy.units`` unit defining the time metric for this body."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value):
        assert value.physical_type == "time"
        self._time_unit = value
        self._time_factor = value.in_units(units.day)

    @property
    def angle_unit(self):
        """An ``astropy.units`` unit defining the angle metric for this body."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, value):
        assert value.physical_type == "angle"
        self._angle_unit = value
        self._angle_factor = value.in_units(units.radian)

    @property
    def _angle_unit(self):
        return self._map._angle_unit

    @_angle_unit.setter
    def _angle_unit(self, value):
        self._map._angle_unit = value

    @property
    def _angle_factor(self):
        return self._map._angle_factor

    @_angle_factor.setter
    def _angle_factor(self, value):
        self._map._angle_factor = value

    @property
    def map(self):
        """The surface map for this body."""
        return self._map

    @map.setter
    def map(self, value):
        assert MapBase in getmro(
            type(value)
        ), "The `map` attribute must be a `starry` map instance."
        assert (
            value._lazy == self._lazy
        ), "Map must have the same evaluation mode (lazy/greedy)."
        self._map = value

    @property
    def r(self):
        """The radius in units of :py:attr:`length_unit`."""
        return self._r / self._length_factor

    @r.setter
    def r(self, value):
        self._r = self._math.cast(value * self._length_factor)

    @property
    def m(self):
        """The mass in units of :py:attr:`mass_unit`."""
        return self._m / self._mass_factor

    @m.setter
    def m(self, value):
        self._m = self._math.cast(value * self._mass_factor)

    @property
    def prot(self):
        """The rotation period in units of :py:attr:`time_unit`."""
        return self._prot / self._time_factor

    @prot.setter
    def prot(self, value):
        self._prot = self._math.cast(value * self._time_factor)

    @property
    def t0(self):
        """A reference time in units of :py:attr:`time_unit`."""
        return self._t0 / self._time_factor

    @t0.setter
    def t0(self, value):
        self._t0 = self._math.cast(value * self._time_factor)

    @property
    def theta0(self):
        """The map rotational phase at time :py:attr:`t0`."""
        return self._theta0 / self._angle_factor

    @theta0.setter
    def theta0(self, value):
        self._theta0 = self._math.cast(value * self._angle_factor)

    def _check_kwargs(self, method, kwargs):
        if not config.quiet:
            for key in kwargs.keys():
                message = "Invalid keyword `{0}` in call to `{1}()`. Ignoring."
                message = message.format(key, method)
                logger.warning(message)


class Primary(Body):
    """A primary (central) body.

    Args:
        map: The surface map of this body. This should be an instance
            returned by :py:func:`starry.Map`.
        r (scalar, optional): The radius of the body in units of
            :py:attr:`length_unit`. Defaults to 1.0.
        m (scalar, optional): The mass of the body in units of
            :py:attr:`mass_unit`. Defaults to 1.0.
        prot (scalar, optional): The rotation period of the body in units of
            :py:attr:`time_unit`. Defaults to 1.0.
        t0 (scalar, optional): A reference time in units of
            :py:attr:`time_unit`. Defaults to 0.0.
        theta0 (scalar, optional): The rotational phase of the map at time
            :py:attr:`t0` in units of :py:attr:`angle_unit`. Defaults to 0.0.
        length_unit (optional): An ``astropy.units`` unit defining the
            distance metric for this object. Defaults to
            :py:attr:`astropy.units.Rsun.`
        mass_unit (optional): An ``astropy.units`` unit defining the
            mass metric for this object. Defaults to
            :py:attr:`astropy.units.Msun.`
        time_unit (optional): An ``astropy.units`` unit defining the
            time metric for this object. Defaults to
            :py:attr:`astropy.units.day.`
        angle_unit (optional): An ``astropy.units`` unit defining the
            angular metric for this object. Defaults to
            :py:attr:`astropy.units.degree.`
    """

    def __init__(self, map, **kwargs):
        # Initialize `Body`
        super(Primary, self).__init__(map, **kwargs)
        for kw in [
            "r",
            "m",
            "prot",
            "t0",
            "theta0",
            "length_unit",
            "mass_unit",
            "time_unit",
            "angle_unit",
        ]:
            kwargs.pop(kw, None)
        self._check_kwargs("Primary", kwargs)


class Secondary(Body):
    """A secondary (orbiting) body.

    Args:
        map: The surface map of this body. This should be an instance
            returned by :py:func:`starry.Map`.
        r (scalar, optional): The radius of the body in units of
            :py:attr:`length_unit`. Defaults to 1.0.
        m (scalar, optional): The mass of the body in units of
            :py:attr:`mass_unit`. Defaults to 1.0.
        a (scalar, optional): The semi-major axis of the body in units of
            :py:attr:`time_unit`. Defaults to 1.0. If :py:attr:`porb` is
            also provided, this value is ignored.
        porb (scalar, optional): The orbital period of the body in units of
            :py:attr:`time_unit`. Defaults to 1.0. Setting this value
            overrides :py:attr:`a`.
        prot (scalar, optional): The rotation period of the body in units of
            :py:attr:`time_unit`. Defaults to 1.0.
        t0 (scalar, optional): A reference time in units of
            :py:attr:`time_unit`. This is taken to be the time of a reference
            transit. Defaults to 0.0.
        ecc (scalar, optional): The orbital eccentricity of the body.
            Defaults to 0.
        w, omega (scalar, optional): The argument of pericenter of the body
            in units of :py:attr:`angle_unit`. Defaults to 90 degrees.
        Omega (scalar, optional): The longitude of ascending node of the
            body in units of :py:attr:`angle_unit`. Defaults to 0 degrees.
        inc (scalar, optional): The orbital inclination of the body in
            units of :py:attr:`angle_unit`. Defaults to 90 degrees.
        theta0 (scalar, optional): The rotational phase of the map at time
            :py:attr:`t0` in units of :py:attr:`angle_unit`. Defaults to
            0.0.
        length_unit (optional): An ``astropy.units`` unit defining the
            distance metric for this object. Defaults to
            :py:attr:`astropy.units.Rsun.`
        mass_unit (optional): An ``astropy.units`` unit defining the
            mass metric for this object. Defaults to
            :py:attr:`astropy.units.Msun.`
        time_unit (optional): An ``astropy.units`` unit defining the
            time metric for this object. Defaults to
            :py:attr:`astropy.units.day.`
        angle_unit (optional): An ``astropy.units`` unit defining the
            angular metric for this object. Defaults to
            :py:attr:`astropy.units.degree.`
    """

    def __init__(self, map, **kwargs):
        # Initialize `Body`
        super(Secondary, self).__init__(map, **kwargs)
        for kw in [
            "r",
            "m",
            "prot",
            "t0",
            "theta0",
            "length_unit",
            "mass_unit",
            "time_unit",
            "angle_unit",
        ]:
            kwargs.pop(kw, None)

        # Attributes
        if kwargs.get("porb", None) is not None:
            self.porb = kwargs.pop("porb", None)
        elif kwargs.get("a", None) is not None:
            self.a = kwargs.pop("a", None)
        else:
            raise ValueError("Must provide a value for either `porb` or `a`.")
        self.ecc = kwargs.pop("ecc", 0.0)
        self.w = kwargs.pop(
            "w", kwargs.pop("omega", 0.5 * np.pi / self._angle_factor)
        )
        self.Omega = kwargs.pop("Omega", 0.0)
        self.inc = kwargs.pop("inc", 0.5 * np.pi / self._angle_factor)
        self._check_kwargs("Secondary", kwargs)

    @property
    def porb(self):
        """The orbital period in units of :py:attr:`time_unit`.

        .. note::
            Setting this value overrides the value of :py:attr:`a`.
        """
        if self._porb == 0.0:
            return None
        else:
            return self._porb / self._time_factor

    @porb.setter
    def porb(self, value):
        self._porb = self._math.cast(value * self._time_factor)
        self._a = 0.0

    @property
    def a(self):
        """The semi-major axis in units of :py:attr:`length_unit`.

        .. note::
            Setting this value overrides the value of :py:attr:`porb`.
        """
        if self._a == 0.0:
            return None
        else:
            return self._a / self._length_factor

    @a.setter
    def a(self, value):
        self._a = self._math.cast(value * self._length_factor)
        self._porb = 0.0

    @property
    def ecc(self):
        """The orbital eccentricity."""
        return self._ecc

    @ecc.setter
    def ecc(self, value):
        self._ecc = value

    @property
    def w(self):
        """The longitude of pericenter in units of :py:attr:`angle_unit`."""
        return self._w / self._angle_factor

    @w.setter
    def w(self, value):
        self._w = self._math.cast(value * self._angle_factor)

    @property
    def omega(self):
        """Alias for the longitude of pericenter :py:attr:`w`."""
        return self.w

    @omega.setter
    def omega(self, value):
        self.w = value

    @property
    def Omega(self):
        """The longitude of ascending node in units of :py:attr:`angle_unit`."""
        return self._Omega / self._angle_factor

    @Omega.setter
    def Omega(self, value):
        self._Omega = self._math.cast(value * self._angle_factor)

    @property
    def inc(self):
        """The orbital inclination in units of :py:attr:`angle_unit`."""
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self._math.cast(value * self._angle_factor)


class System(object):
    """A system of bodies in Keplerian orbits about a central primary body.

    Args:
        primary (:py:class:`Primary`): The central body.
        secondaries (:py:class:`Secondary`): One or more secondary bodies
            in orbit about the primary.
        time_unit (optional): An ``astropy.units`` unit defining the
            time metric for this object. Defaults to
            :py:attr:`astropy.units.day.`
        light_delay (bool, optional): Account for the light travel time
            delay to the barycenter of the system? Default is False.
        texp (scalar): The exposure time of each observation. This can be a
            scalar or a tensor with the same shape as ``t``. If ``texp`` is
            provided, ``t`` is assumed to indicate the timestamp at the middle
            of an exposure of length ``texp``.
        oversample (int): The number of function evaluations to use when
            numerically integrating the exposure time.
        order (int): The order of the numerical integration scheme. This must
            be one of the following: ``0`` for a centered Riemann sum
            (equivalent to the "resampling" procedure suggested by Kipping 2010),
            ``1`` for the trapezoid rule, or ``2`` for Simpson’s rule.
    """

    def _no_spectral(self):
        if self._primary._map.nw is not None:  # pragma: no cover
            raise NotImplementedError(
                "Method not yet implemented for spectral maps."
            )

    def __init__(
        self,
        primary,
        *secondaries,
        time_unit=units.day,
        light_delay=False,
        texp=None,
        oversample=7,
        order=0,
    ):
        # Units
        self.time_unit = time_unit
        self._light_delay = bool(light_delay)
        if texp is None:
            self._texp = 0.0
        else:
            self._texp = texp
        assert self._texp >= 0.0, "Parameter `texp` must be >= 0."
        self._oversample = int(oversample)
        assert self._oversample > 0, "Parameter `oversample` must be > 0."
        self._order = int(order)
        assert self._order in [0, 1, 2], "Invalid value for parameter `order`."

        # Primary body
        assert (
            type(primary) is Primary
        ), "Argument `primary` must be an instance of `Primary`."
        assert (
            primary._map.__props__["reflected"] == False
        ), "Reflected light map not allowed for the primary body."
        self._primary = primary
        self._rv = primary._map.__props__["rv"]
        self._lazy = primary._lazy
        self._math = primary._math
        if self._lazy:
            self._linalg = math.lazy_linalg
        else:
            self._linalg = math.greedy_linalg

        # Secondary bodies
        assert len(secondaries) > 0, "There must be at least one secondary."
        for sec in secondaries:
            assert type(sec) is Secondary, (
                "Argument `*secondaries` must be a sequence of "
                "`Secondary` instances."
            )
            assert (
                sec._map.nw == self._primary._map.nw
            ), "All bodies must have the same number of wavelength bins `nw`."
            assert sec._map.__props__["rv"] == self._rv, (
                "Radial velocity must be enabled "
                "for either all or none of the bodies."
            )
            assert (
                sec._lazy == self._lazy
            ), "All bodies must have the same evaluation mode (lazy/greedy)."

        reflected = [sec._map.__props__["reflected"] for sec in secondaries]
        if np.all(reflected):
            self._reflected = True
        elif np.any(reflected):
            raise ValueError(
                "Reflected light must be enabled "
                "for either all or none of the secondaries."
            )
        else:
            self._reflected = False
        self._secondaries = secondaries

        # All bodies
        self._bodies = [self._primary] + list(self._secondaries)

        # Indices of each of the bodies in the design matrix
        Ny = [self._primary._map.Ny] + [
            sec._map.Ny for sec in self._secondaries
        ]
        self._inds = []
        cur = 0
        for N in Ny:
            self._inds.append(cur + np.arange(N))
            cur += N

        # Theano ops class
        self.ops = OpsSystem(
            self._primary,
            self._secondaries,
            reflected=self._reflected,
            rv=self._rv,
            light_delay=self._light_delay,
            texp=self._texp,
            oversample=self._oversample,
            order=self._order,
        )

        # Solve stuff
        self._flux = None
        self._C = None
        self._solution = None
        self._solved_bodies = []

    @property
    def light_delay(self):
        """Account for the light travel time delay? *Read-only*"""
        return self._light_delay

    @property
    def texp(self):
        """The exposure time in units of :py:attr:`time_unit`. *Read-only*"""

    @property
    def oversample(self):
        """Oversample factor when integrating over exposure time. *Read-only*"""
        return self._oversample

    @property
    def order(self):
        """The order of the numerical integration scheme. *Read-only*

        - ``0``: a centered Riemann sum
        - ``1``: trapezoid rule
        - ``2``: Simpson’s rule
        """
        return self._order

    @property
    def time_unit(self):
        """An ``astropy.units`` unit defining the time metric for the system."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value):
        assert value.physical_type == "time"
        self._time_unit = value
        self._time_factor = value.in_units(units.day)

    @property
    def primary(self):
        """The primary (central) object in the Keplerian system."""
        return self._primary

    @property
    def secondaries(self):
        """A list of the secondary (orbiting) object(s) in the Keplerian system."""
        return self._secondaries

    @property
    def bodies(self):
        """A list of all objects in the Keplerian system."""
        return self._bodies

    @property
    def map_indices(self):
        """A list of the indices corresponding to each body in the design matrix."""
        return self._inds

    def show(
        self,
        t,
        cmap="plasma",
        res=300,
        interval=75,
        file=None,
        figsize=(3, 3),
        html5_video=True,
        window_pad=1.0,
        **kwargs,
    ):
        """Visualize the Keplerian system.

        Note that the body surface intensities are not normalized.

        Args:
            t (scalar or vector): The time(s) at which to evaluate the orbit and
                the map in units of :py:attr:`time_unit`.
            cmap (string or colormap instance, optional): The matplotlib colormap
                to use. Defaults to ``plasma``.
            res (int, optional): The resolution of the map in pixels on a
                side. Defaults to 300.
            figsize (tuple, optional): Figure size in inches. Default is
                (3, 3) for orthographic maps and (7, 3.5) for rectangular
                maps.
            interval (int, optional): Interval between frames in milliseconds
                (animated maps only). Defaults to 75.
            file (string, optional): The file name (including the extension)
                to save the animation to (animated maps only). Defaults to None.
            html5_video (bool, optional): If rendering in a Jupyter notebook,
                display as an HTML5 video? Default is True. If False, displays
                the animation using Javascript (file size will be larger.)
            window_pad (float, optional): Padding around the primary in units
                of the primary radius. Bodies outside of this window will be
                cropped. Default is 1.0.

        .. note::
            If calling this method on an instance of ``System`` created within
            a ``pymc3.Model()``, you may specify a ``point`` keyword with
            the model point at which to evaluate the map. This method also
            accepts a ``model`` keyword, although this is inferred
            automatically if called from within a ``pymc3.Model()`` context.
            If no point is provided, attempts to evaluate the map at
            ``model.test_point`` and raises a warning.

        """
        # Not yet implemented
        if self._primary._map.nw is not None:  # pragma: no cover
            raise NotImplementedError(
                "Method not implemented for spectral maps."
            )

        # So we can evaluate stuff in lazy mode
        get_val = evaluator(**kwargs)

        # Render the maps & get the orbital positions
        if self._rv:
            self._primary.map._set_RV_filter()
            for sec in self._secondaries:
                sec.map._set_RV_filter()
        img_pri, img_sec, x, y, z = self.ops.render(
            self._math.reshape(self._math.to_array_or_tensor(t), [-1])
            * self._time_factor,
            res,
            self._primary._r,
            self._primary._m,
            self._primary._prot,
            self._primary._t0,
            self._primary._theta0,
            self._primary._map._inc,
            self._primary._map._obl,
            self._primary._map._y,
            self._primary._map._u,
            self._primary._map._f,
            self._math.to_array_or_tensor(
                [sec._r for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._m for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._prot for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._t0 for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._theta0 for sec in self._secondaries]
            ),
            self._get_periods(),
            self._math.to_array_or_tensor(
                [sec._ecc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._w for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._Omega for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._inc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._inc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._obl for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._y for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._u for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._f for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._sigr for sec in self._secondaries]
            ),
        )

        # Convert to units of the primary radiu
        x, y, z = (
            x / self._primary._r,
            y / self._primary._r,
            z / self._primary._r,
        )
        r = self._math.to_array_or_tensor(
            [sec._r for sec in self._secondaries]
        )
        r = r / self._primary._r

        # Evaluate if needed
        if self._lazy:
            img_pri = get_val(img_pri)
            img_sec = get_val(img_sec)
            x = get_val(x)
            y = get_val(y)
            z = get_val(z)
            r = get_val(r)

        # We need this to be of shape (nplanet, nframe)
        x = x.T
        y = y.T
        z = z.T

        # Ensure we have an array of frames
        if len(img_pri.shape) == 3:
            nframes = img_pri.shape[0]
        else:  # pragma: no cover
            nframes = 1
            img_pri = np.reshape(img_pri, (1,) + img_pri.shape)
            img_sec = np.reshape(img_sec, (1,) + img_sec.shape)
        animated = nframes > 1

        # Set up the plot
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.axis("off")
        ax.set_xlim(-1.0 - window_pad, 1.0 + window_pad)
        ax.set_ylim(-1.0 - window_pad, 1.0 + window_pad)

        # Render the first frame
        img = [None for n in range(1 + len(self._secondaries))]
        circ = [None for n in range(1 + len(self._secondaries))]
        extent = np.array([-1.0, 1.0, -1.0, 1.0])
        img[0] = ax.imshow(
            img_pri[0],
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="none",
            vmin=np.nanmin(img_pri),
            vmax=np.nanmax(img_pri),
            animated=animated,
            zorder=0.0,
        )
        circ[0] = plt.Circle(
            (0, 0), 1, color="k", fill=False, zorder=1e-3, lw=2
        )
        ax.add_artist(circ[0])
        for i, _ in enumerate(self._secondaries):
            extent = np.array([x[i, 0], x[i, 0], y[i, 0], y[i, 0]]) + (
                r[i] * np.array([-1.0, 1.0, -1.0, 1.0])
            )
            img[i + 1] = ax.imshow(
                img_sec[i, 0],
                origin="lower",
                extent=extent,
                cmap=cmap,
                interpolation="none",
                vmin=np.nanmin(img_sec),
                vmax=np.nanmax(img_sec),
                animated=animated,
                zorder=z[i, 0],
            )
            circ[i] = plt.Circle(
                (x[i, 0], y[i, 0]),
                r[i],
                color="k",
                fill=False,
                zorder=z[i, 0] + 1e-3,
                lw=2,
            )
            ax.add_artist(circ[i])

        # Animation
        if animated:

            def updatefig(k):

                # Update Primary map
                img[0].set_array(img_pri[k])

                # Update Secondary maps & positions
                for i, _ in enumerate(self._secondaries):
                    extent = np.array([x[i, k], x[i, k], y[i, k], y[i, k]]) + (
                        r[i] * np.array([-1.0, 1.0, -1.0, 1.0])
                    )
                    if np.any(np.abs(extent) < 1.0 + window_pad):
                        img[i + 1].set_array(img_sec[i, k])
                        img[i + 1].set_extent(extent)
                        img[i + 1].set_zorder(z[i, k])
                        circ[i].center = (x[i, k], y[i, k])
                        circ[i].set_zorder(z[i, k] + 1e-3)

                return img + circ

            ani = FuncAnimation(
                fig, updatefig, interval=interval, blit=False, frames=nframes
            )

            # Business as usual
            if (file is not None) and (file != ""):
                if file.endswith(".mp4"):
                    ani.save(file, writer="ffmpeg")
                elif file.endswith(".gif"):
                    ani.save(file, writer="imagemagick")
                else:  # pragma: no cover
                    # Try and see what happens!
                    ani.save(file)
                plt.close()
            else:  # pragma: no cover
                try:
                    if "zmqshell" in str(type(get_ipython())):
                        plt.close()
                        if html5_video:
                            display(HTML(ani.to_html5_video()))
                        else:
                            display(HTML(ani.to_jshtml()))
                    else:
                        raise NameError("")
                except NameError:
                    plt.show()
                    plt.close()

            # Matplotlib generates an annoying empty
            # file when producing an animation. Delete it.
            try:
                os.remove("None0000000.png")
            except FileNotFoundError:
                pass

        else:

            if (file is not None) and (file != ""):
                fig.savefig(file)
                plt.close()
            else:  # pragma: no cover
                plt.show()

        if self._rv:
            self._primary.map._unset_RV_filter()
            for sec in self._secondaries:
                sec.map._unset_RV_filter()

    def design_matrix(self, t):
        """Compute the system flux design matrix at times ``t``.

        .. note::

            This is the *unweighted* design matrix, i.e., it does not
            include the scaling by the amplitude of each body's map.
            To perform this weighting, do

            .. code-block:: python

                X = sys.design_matrix(**kwargs)
                for i, body in zip(sys.map_indices, sys.bodies):
                    X[:, i] *= body.map.amp

        Args:
            t (scalar or vector): An array of times at which to evaluate
                the design matrix in units of :py:attr:`time_unit`.
        """
        return self.ops.X(
            self._math.reshape(self._math.to_array_or_tensor(t), [-1])
            * self._time_factor,
            self._primary._r,
            self._primary._m,
            self._primary._prot,
            self._primary._t0,
            self._primary._theta0,
            self._math.to_array_or_tensor(1.0),
            self._primary._map._inc,
            self._primary._map._obl,
            self._primary._map._u,
            self._primary._map._f,
            self._math.to_array_or_tensor(
                [sec._r for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._m for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._prot for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._t0 for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._theta0 for sec in self._secondaries]
            ),
            self._get_periods(),
            self._math.to_array_or_tensor(
                [sec._ecc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._w for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._Omega for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._inc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [
                    self._math.to_array_or_tensor(1.0)
                    for sec in self._secondaries
                ]
            ),
            self._math.to_array_or_tensor(
                [sec._map._inc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._obl for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._u for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._f for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._sigr for sec in self._secondaries]
            ),
        )

    def flux(self, t, total=True):
        """Compute the system flux at times ``t``.

        Args:
            t (scalar or vector): An array of times at which to evaluate
                the flux in units of :py:attr:`time_unit`.
            total (bool, optional): Return the total system flux? Defaults to
                True. If False, returns arrays corresponding to the flux
                from each body.
        """
        X = self.design_matrix(t)

        # Weight the ylms by amplitude
        if self._reflected:
            # If we're doing reflected light, scale the amplitude of
            # each of the secondaries by the amplitude of the primary
            # (the illumination source).
            ay = [self._primary.map.amp * self._primary._map._y] + [
                self._primary.map.amp * body.map.amp * body._map._y
                for body in self._secondaries
            ]
        else:
            ay = [body.map.amp * body._map._y for body in self._bodies]

        if total:
            return self._math.dot(X, self._math.concatenate(ay))
        else:
            return [
                self._math.dot(X[:, idx], ay[i])
                for i, idx in enumerate(self._inds)
            ]

    def rv(self, t, keplerian=True, total=True):
        """Compute the observed radial velocity of the system at times ``t``.

        Args:
            t (scalar or vector): An array of times at which to evaluate
                the radial velocity in units of :py:attr:`time_unit`.
            keplerian (bool): Include the Keplerian component of the radial
                velocity of the primary? Default is True. If False, this
                method returns a model for only the radial velocity anomaly
                due to transits (the Rossiter-McLaughlin effect) and
                time-variable surface features (Doppler tomography) for all
                bodies in the system.
            total (bool, optional): Return the total system RV? Defaults to
                True. If False, returns arrays corresponding to the RV
                contribution from each body.

        Returns:
            A vector or matrix of radial velocities in units of meters per
            second. If ``total`` is ``True``, this returns a vector of the
            total stellar radial velocity at every point ``t``. This is
            the sum of all effects: the Keplerian motion of the star due to
            the planets, the radial velocity anomaly due to spots or
            features on its surface rotating in and out of view, as well
            as the Rossiter-McLaughlin effect due to transits by the
            secondaries.
            If ``total`` is ``False``, this returns a matrix whose
            rows are the radial velocity contributions to the measured **stellar**
            radial velocity from each of the bodies in the system.
            As a specific example, if there are three bodies in the
            system (one ``Primary`` and two ``Secondary`` bodies), the
            first row is the radial velocity anomaly corresponding to the star's
            own Doppler signals (due to spots rotating in and out of
            view, or due to the Rossiter-McLaughlin effect of the other
            two bodies transiting it). The other two rows are the Keplerian
            contribution from each of the ``Secondary`` bodies, plus any
            radial velocity anomaly due to spots or occultations of their
            surfaces.
            The sum across all rows is equal to the total radial velocity
            and is the same as what this method would return if ``total=True``.

        Note, importantly, that when ``total=False``, this method does
        *not* return the radial velocities of each of the bodies; instead,
        it returns the *contribution to the stellar radial velocity* due
        to each of the bodies. If you require knowing the radial velocity
        of the secondary objects, you can compute this from conservation
        of momentum:

        .. code-block::python

            # Instantiate a system w/ a star and two planets
            A = starry.Primary(...)
            b = starry.Secondary(...)
            c = starry.Secondary(...)
            sys = starry.System(A, b, c)

            # Get the contribution from each body to the star's RV
            rv = sys.rv(t, total=False)

            # Conservation of momentum implies the RV of `b`
            # is proportional to the RV of the star, weighted
            # by the mass ratio. We can compute the mass ratio
            # if we're mindful of the (potentially different)
            # units for the star and the planets:
            mA_mb = ((A.m * A.mass_unit) / (b.m * b.mass_unit)).decompose()
            rv_b = -rv[1] * mA_mb

            # Same for planet `c`
            mA_mc = ((A.m * A.mass_unit) / (b.m * b.mass_unit)).decompose()
            rv_c = -rv[2] * mAmc

        Note that this method implicitly assumes multi-Keplerian orbits;
        i.e., the ``Secondary`` bodies are treated as massive *only* when computing
        their gravitational effect on the ``Primary`` (as opposed to each other).
        This therefore ignores all ``Secondary``-``Secondary``
        (i.e., planet-planet) interactions.

        """
        assert self._rv, "Only implemented if `rv=True` for all body maps."
        rv = self.ops.rv(
            self._math.reshape(self._math.to_array_or_tensor(t), [-1])
            * self._time_factor,
            self._primary._r,
            self._primary._m,
            self._primary._prot,
            self._primary._t0,
            self._primary._theta0,
            self._primary._map._amp,
            self._primary._map._inc,
            self._primary._map._obl,
            self._primary._map._y,
            self._primary._map._u,
            self._primary._map._alpha,
            self._primary._map._veq,
            self._math.to_array_or_tensor(
                [sec._r for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._m for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._prot for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._t0 for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._theta0 for sec in self._secondaries]
            ),
            self._get_periods(),
            self._math.to_array_or_tensor(
                [sec._ecc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._w for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._Omega for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._inc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._amp for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._inc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._obl for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._y for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._u for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._alpha for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._sigr for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._map._veq for sec in self._secondaries]
            ),
            np.array(keplerian),
        )
        if total:
            return self._math.sum(rv, axis=0)
        else:
            return rv

    def position(self, t):
        """Compute the Cartesian positions of all bodies at times ``t``.

        Args:
            t (scalar or vector): An array of times at which to evaluate
                the position in units of :py:attr:`time_unit`.
        """
        x, y, z = self.ops.position(
            self._math.reshape(self._math.to_array_or_tensor(t), [-1])
            * self._time_factor,
            self._primary._m,
            self._primary._t0,
            self._math.to_array_or_tensor(
                [sec._m for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._t0 for sec in self._secondaries]
            ),
            self._get_periods(),
            self._math.to_array_or_tensor(
                [sec._ecc for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._w for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._Omega for sec in self._secondaries]
            ),
            self._math.to_array_or_tensor(
                [sec._inc for sec in self._secondaries]
            ),
        )
        fac = np.reshape(
            [self._primary._length_factor]
            + [sec._length_factor for sec in self._secondaries],
            [-1, 1],
        )
        return (x / fac, y / fac, z / fac)

    def _get_periods(self):
        periods = [None for sec in self._secondaries]
        for i, sec in enumerate(self._secondaries):
            if sec._porb:
                periods[i] = sec._porb
            else:
                periods[i] = (
                    (2 * np.pi)
                    * sec._a ** (3 / 2)
                    / (self._math.sqrt(G_grav * (self._primary._m + sec._m)))
                )
        return self._math.to_array_or_tensor(periods)

    def set_data(self, flux, C=None, cho_C=None):
        """Set the data vector and covariance matrix.

        This method is required by the :py:meth:`solve` method, which
        analytically computes the posterior over surface maps for all bodies
        in the system given a dataset and a prior, provided both are described
        as multivariate Gaussians.

        Args:
            flux (vector): The observed system light curve.
            C (scalar, vector, or matrix): The data covariance. This may be
                a scalar, in which case the noise is assumed to be
                homoscedastic, a vector, in which case the covariance
                is assumed to be diagonal, or a matrix specifying the full
                covariance of the dataset. Default is None. Either `C` or
                `cho_C` must be provided.
            cho_C (matrix): The lower Cholesky factorization of the data
                covariance matrix. Defaults to None. Either `C` or
                `cho_C` must be provided.
        """
        self._flux = self._math.cast(flux)
        self._C = self._linalg.Covariance(
            C=C, cho_C=cho_C, N=self._flux.shape[0]
        )

    def solve(self, *, design_matrix=None, t=None):
        """Solve the least-squares problem for the posterior over maps for all bodies.

        This method solves the generalized least squares problem given a system
        light curve and its covariance (set via the :py:meth:`set_data` method)
        and a Gaussian prior on the spherical harmonic coefficients
        (set via the :py:meth:`set_prior` method). The map amplitudes and
        coefficients of each of the bodies in the system are then set to the
        maximum a posteriori (MAP) solution.

        Args:
            design_matrix (matrix, optional): The flux design matrix, the
                quantity returned by :py:meth:`design_matrix`. Default is
                None, in which case this is computed based on ``kwargs``.
            t (vector, optional): The vector of times at which to evaluate
                :py:meth:`design_matrix`, if a design matrix is not provided.
                Default is None.

        Returns:
            The posterior mean for the spherical harmonic \
            coefficients `l > 0` and the Cholesky factorization of the \
            posterior covariance of all of the bodies in the system, \
            stacked in order (primary, followed by each of the secondaries \
            in the order they were provided.)

        .. note::
            Users may call the :py:meth:`draw` method of this class to draw
            from the posterior after calling :py:meth:`solve`.
        """
        # TODO: Implement for spectral maps?
        self._no_spectral()

        # Check that the data is set
        if self._flux is None or self._C is None:
            raise ValueError("Please provide a dataset with `set_data()`.")

        # Get the full design matrix
        if design_matrix is None:
            assert t is not None, "Please provide a time vector `t`."
            design_matrix = self.design_matrix(t)
        X = self._math.cast(design_matrix)

        # Get the data vector
        f = self._math.cast(self._flux)

        # Check for bodies whose priors are set
        self._solved_bodies = []
        inds = []
        dense_L = False
        for k, body in enumerate(self._bodies):

            if body.map._mu is None or body.map._L is None:

                # Subtract out this term from the data vector,
                # since it is fixed
                f -= body.map.amp * self._math.dot(
                    X[:, self._inds[k]], body.map.y
                )

            else:

                # Add to our list of indices/bodies to solve for
                inds.extend(self._inds[k])
                self._solved_bodies.append(body)
                if body.map._L.kind in ["matrix", "cholesky"]:
                    dense_L = True

        # Do we have at least one body?
        if len(self._solved_bodies) == 0:
            raise ValueError("Please provide a prior for at least one body.")

        # Keep only the terms we'll solve for
        X = X[:, inds]

        # Stack our priors
        mu = self._math.concatenate(
            [body.map._mu for body in self._solved_bodies]
        )

        if not dense_L:
            # We can just concatenate vectors
            LInv = self._math.concatenate(
                [
                    body.map._L.inverse * self._math.ones(body.map.Ny)
                    for body in self._solved_bodies
                ]
            )
        else:
            # FACT: The inverse of a block diagonal matrix
            # is the block diagonal matrix of the inverses.
            LInv = self._math.block_diag(
                *[
                    body.map._L.inverse * self._math.eye(body.map.Ny)
                    for body in self._solved_bodies
                ]
            )

        # Compute the MAP solution
        self._solution = self._linalg.solve(X, f, self._C.cholesky, mu, LInv)

        # Set all the map vectors
        x, cho_cov = self._solution
        n = 0
        for body in self._solved_bodies:
            inds = slice(n, n + body.map.Ny)
            body.map.amp = x[inds][0]
            if body.map.ydeg > 0:
                body.map[1:, :] = x[inds][1:] / body.map.amp
            n += body.map.Ny

        # Return the mean and covariance
        self._solution = (x, cho_cov)
        return self._solution

    @property
    def solution(self):
        r"""The posterior probability distribution for the maps in the system.

        This is a tuple containing the mean and lower Cholesky factorization of the
        covariance of the amplitude-weighted spherical harmonic coefficient vectors,
        obtained by solving the regularized least-squares problem
        via the :py:meth:`solve` method.

        Note that to obtain the actual covariance matrix from the lower Cholesky
        factorization :math:`L`, simply compute :math:`L L^\top`.

        Note also that this is the posterior for the **amplitude-weighted**
        map vectors. Under this convention, the map amplitude is equal to the
        first term of the vector of each body and the spherical harmonic coefficients are
        equal to the vector normalized by the first term.
        """
        if self._solution is None:
            raise ValueError("Please call `solve()` first.")
        return self._solution

    def draw(self):
        """
        Draw a map from the posterior distribution and set
        the :py:attr:`y` map vector of each body.

        Users should call :py:meth:`solve` to enable this attribute.
        """
        if self._solution is None:
            raise ValueError("Please call `solve()` first.")

        # Number of coefficients
        N = np.sum([body.map.Ny for body in self._solved_bodies])

        # Fast multivariate sampling using the Cholesky factorization
        yhat, cho_ycov = self._solution
        u = self._math.cast(np.random.randn(N))
        x = yhat + self._math.dot(cho_ycov, u)

        # Set all the map vectors
        n = 0
        for body in self._solved_bodies:
            inds = slice(n, n + body.map.Ny)
            body.map.amp = x[inds][0]
            body.map[1:, :] = x[inds][1:] / body.map.amp
            n += body.map.Ny

    def lnlike(self, *, design_matrix=None, t=None, woodbury=True):
        """Returns the log marginal likelihood of the data given a design matrix.

        This method computes the marginal likelihood (marginalized over the
        spherical harmonic coefficients of all bodies) given a system
        light curve and its covariance (set via the :py:meth:`set_data` method)
        and a Gaussian prior on the spherical harmonic coefficients
        (set via the :py:meth:`set_prior` method).

        Args:
            design_matrix (matrix, optional): The flux design matrix, the
                quantity returned by :py:meth:`design_matrix`. Default is
                None, in which case this is computed based on ``kwargs``.
            t (vector, optional): The vector of times at which to evaluate
                :py:meth:`design_matrix`, if a design matrix is not provided.
                Default is None.
            woodbury (bool, optional): Solve the linear problem using the
                Woodbury identity? Default is True. The
                `Woodbury identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_
                is used to speed up matrix operations in the case that the
                number of data points is much larger than the number of
                spherical harmonic coefficients. In this limit, it can
                speed up the code by more than an order of magnitude. Keep
                in mind that the numerical stability of the Woodbury identity
                is not great, so if you're getting strange results try
                disabling this. It's also a good idea to disable this in the
                limit of few data points and large spherical harmonic degree.

        Returns:
            lnlike: The log marginal likelihood.
        """
        # TODO: Implement for spectral maps?
        self._no_spectral()

        # Check that the data is set
        if self._flux is None or self._C is None:
            raise ValueError("Please provide a dataset with `set_data()`.")

        # Get the full design matrix
        if design_matrix is None:
            assert t is not None, "Please provide a time vector `t`."
            design_matrix = self.design_matrix(t)
        X = self._math.cast(design_matrix)

        # Get the data vector
        f = self._math.cast(self._flux)

        # Check for bodies whose priors are set
        self._solved_bodies = []
        inds = []
        dense_L = False
        for k, body in enumerate(self._bodies):

            if body.map._mu is None or body.map._L is None:

                # Subtract out this term from the data vector,
                # since it is fixed
                f -= body.map.amp * self._math.dot(
                    X[:, self._inds[k]], body.map.y
                )

            else:

                # Add to our list of indices/bodies to solve for
                inds.extend(self._inds[k])
                self._solved_bodies.append(body)
                if body.map._L.kind in ["matrix", "cholesky"]:
                    dense_L = True

        # Do we have at least one body?
        if len(self._solved_bodies) == 0:
            raise ValueError("Please provide a prior for at least one body.")

        # Keep only the terms we'll solve for
        X = X[:, inds]

        # Stack our priors
        mu = self._math.concatenate(
            [body.map._mu for body in self._solved_bodies]
        )

        # Compute the likelihood
        if woodbury:
            if not dense_L:
                # We can just concatenate vectors
                LInv = self._math.concatenate(
                    [
                        body.map._L.inverse * self._math.ones(body.map.Ny)
                        for body in self._solved_bodies
                    ]
                )
            else:
                LInv = self._math.block_diag(
                    *[
                        body.map._L.inverse * self._math.eye(body.map.Ny)
                        for body in self._solved_bodies
                    ]
                )
            lndetL = self._math.cast(
                [body.map._L.lndet for body in self._solved_bodies]
            )
            return self._linalg.lnlike_woodbury(
                X, f, self._C.inverse, mu, LInv, self._C.lndet, lndetL
            )
        else:
            if not dense_L:
                # We can just concatenate vectors
                L = self._math.concatenate(
                    [
                        body.map._L.value * self._math.ones(body.map.Ny)
                        for body in self._solved_bodies
                    ]
                )
            else:
                L = self._math.block_diag(
                    *[
                        body.map._L.value * self._math.eye(body.map.Ny)
                        for body in self._solved_bodies
                    ]
                )
            return self._linalg.lnlike(X, f, self._C.value, mu, L)
