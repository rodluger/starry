import numpy as np
from functools import wraps
from warnings import warn

__all__ = ["YlmBase"]


def count_spaces(docstring):
    """
    Hacky function to figure out the minimum indentation
    level of a docstring.

    """
    lines = docstring.split("\n")
    for line in lines:
        if line.startswith(" "):
            for t in range(len(line)):
                if line[t] != " ":
                    return t
    return 0


def deprecated(replaced_by=None, extra_info=None, version=None):
    """
    Wrap the method `func` and warn the user it is deprecated.

    """

    def outer_wrapper(method):

        # Deprecation message
        message = "Method ``{}`` is deprecated{} and will be removed in the future.".format(
            method.__name__,
            " as of version ``{}``".format(version)
            if version is not None
            else "",
        )
        if replaced_by is not None:
            message += " Please use method ``{}`` instead.".format(replaced_by)
        if extra_info is not None:
            message += " For more information, see {}.".format(extra_info)

        @wraps(method)  # inherit docstring
        def inner_wrapper(instance, *args, **kwargs):

            # Raise the warning
            warn(message, DeprecationWarning)

            # Evaluate the method
            return method(instance, *args, **kwargs)

        # Amend the docstring
        tabs = " " * count_spaces(inner_wrapper.__doc__)
        inner_wrapper.__doc__ += "\n{tabs}.. warning::\n\n{tabs}    {message}".format(
            message=message, tabs=tabs
        )

        return inner_wrapper

    return outer_wrapper


class YlmBase:
    @deprecated(replaced_by="spot", version="1.1")
    def add_spot(
        self,
        amp=None,
        intensity=None,
        relative=True,
        sigma=0.1,
        lat=0.0,
        lon=0.0,
    ):
        r"""Add the expansion of a gaussian spot to the map.

        This function adds a spot whose functional form is the spherical
        harmonic expansion of a gaussian in the quantity
        :math:`\cos\Delta\theta`, where :math:`\Delta\theta`
        is the angular separation between the center of the spot and another
        point on the surface. The spot brightness is controlled by either the
        parameter ``amp``, defined as the fractional change in the
        total luminosity of the object due to the spot, or the parameter
        ``intensity``, defined as the fractional change in the
        intensity at the center of the spot.

        Args:
            amp (scalar or vector, optional): The amplitude of the spot. This
                is equal to the fractional change in the luminosity of the map
                due to the spot. If the map has more than one wavelength bin,
                this must be a vector of length equal to the number of
                wavelength bins. Default is None.
                Either ``amp`` or ``intensity`` must be given.
            intensity (scalar or vector, optional): The intensity of the spot.
                This is equal to the fractional change in the intensity of the
                map at the *center* of the spot. If the map has more than one
                wavelength bin, this must be a vector of length equal to the
                number of wavelength bins. Default is None.
                Either ``amp`` or ``intensity`` must be given.
            relative (bool, optional): If True, computes the spot expansion
                assuming the fractional `amp` or `intensity` change is relative
                to the **current** map amplitude/intensity. If False, computes
                the spot expansion assuming the fractional change is relative
                to the **original** map amplitude/intensity (i.e., that of
                a featureless map). Defaults to True. Note that if True,
                adding two spots with the same values of `amp` or `intensity`
                will generally result in *different* intensities at their
                centers, since the first spot will have changed the map
                intensity everywhere! Defaults to True.
            sigma (scalar, optional): The standard deviation of the gaussian.
                Defaults to 0.1.
            lat (scalar, optional): The latitude of the spot in units of
                :py:attr:`angle_unit`. Defaults to 0.0.
            lon (scalar, optional): The longitude of the spot in units of
                :py:attr:`angle_unit`. Defaults to 0.0.

        """
        # Parse the amplitude
        if (amp is None and intensity is None) or (
            amp is not None and intensity is not None
        ):
            raise ValueError("Please provide either `amp` or `intensity`.")
        elif amp is not None:
            amp, _ = self._math.vectorize(
                self._math.cast(amp), np.ones(self.nw)
            )
            # Normalize?
            if not relative:
                amp /= self.amp
        else:
            # Vectorize if needed
            intensity, _ = self._math.vectorize(
                self._math.cast(intensity), np.ones(self.nw)
            )
            # Normalize?
            if not relative:
                baseline = 1.0 / np.pi
            else:
                baseline = self.intensity(lat=lat, lon=lon, limbdarken=False)
            DeltaI = baseline * intensity
            # The integral of the gaussian in cos(Delta theta) over the
            # surface of the sphere is sigma * sqrt(2 * pi^3). Combining
            # this with the normalization convention of starry (a factor of 4),
            # the corresponding spot amplitude is...
            amp = sigma * np.sqrt(2 * np.pi ** 3) * DeltaI / 4
            if not relative:
                amp /= self.amp

        # Parse remaining kwargs
        sigma, lat, lon = self._math.cast(sigma, lat, lon)

        # Get the Ylm expansion of the spot. Note that yspot[0] is not
        # unity, so we'll need to normalize it before setting self._y
        yspot = self.ops.expand_spot(
            amp, sigma, lat * self._angle_factor, lon * self._angle_factor
        )
        y_new = self._y + yspot
        amp_new = self._amp * y_new[0]
        y_new /= y_new[0]

        # Update the map and the normalizing amplitude
        self._y = y_new
        self._amp = amp_new
