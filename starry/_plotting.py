# -*- coding: utf-8 -*-
from ._constants import *
import numpy as np


__all__ = [
    "get_ortho_latitude_lines",
    "get_ortho_longitude_lines",
    "get_moll_latitude_lines",
    "get_moll_longitude_lines",
    "get_projection",
]


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3],
    )


def get_moll_latitude_lines(dlat=np.pi / 6, npts=1000, niter=100):
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:
        theta = lat
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), npts)
        y = np.ones(npts) * np.sqrt(2) * np.sin(theta)
        a = np.sqrt(2)
        b = 2 * np.sqrt(2)
        y[(y / a) ** 2 + (x / b) ** 2 > 1] = np.nan
        res.append((x, y))
    return res


def get_moll_longitude_lines(dlon=np.pi / 6, npts=1000, niter=100):
    res = []
    lonlines = np.arange(-np.pi, np.pi, dlon)[1:]
    for lon in lonlines:
        lat = np.linspace(-np.pi / 2, np.pi / 2, npts)
        theta = np.array(lat)
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
        y = np.sqrt(2) * np.sin(theta)
        res.append((x, y))
    return res


def get_ortho_latitude_lines(inc=np.pi / 2, obl=0, dlat=np.pi / 6, npts=1000):
    """

    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Latitude lines
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if si != 0:
            if inc > np.pi / 2:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 < ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 < ymax] = np.nan
            else:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 > ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 > ymax] = np.nan

        # Rotate them
        for y in (y1, y2):
            xr = -x * co - y * so
            yr = -x * so + y * co
            res.append((xr, yr))

    return res


def get_ortho_longitude_lines(
    inc=np.pi / 2, obl=0, theta=0, dlon=np.pi / 6, npts=1000
):
    """

    """

    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Are we (essentially) equator-on?
    equator_on = (inc > 88 * np.pi / 180) and (inc < 92 * np.pi / 180)

    # Longitude grid lines
    res = []
    if equator_on:
        offsets = np.arange(-np.pi / 2, np.pi / 2, dlon)
    else:
        offsets = np.arange(0, 2 * np.pi, dlon)

    for offset in offsets:

        # Super hacky, sorry. This can probably
        # be coded up more intelligently.
        if equator_on:
            sgns = [1]
            if np.cos(theta + offset) >= 0:
                bsgn = 1
            else:
                bsgn = -1
        else:
            bsgn = 1
            if np.cos(theta + offset) >= 0:
                sgns = np.array([1, -1])
            else:
                sgns = np.array([-1, 1])

        for lon, sgn in zip([0, np.pi], sgns):

            # Viewed at i = 90
            y = np.linspace(-1, 1, npts)
            b = bsgn * np.sin(lon - theta - offset)
            x = b * np.sqrt(1 - y ** 2)
            z = sgn * np.sqrt(np.abs(1 - x ** 2 - y ** 2))

            if equator_on:

                pass

            else:

                # Rotate by the inclination
                R = RAxisAngle([1, 0, 0], np.pi / 2 - inc)
                v = np.vstack(
                    (x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1))
                )
                x, y, _ = np.dot(R, v)

                # Mask lines on the backside
                if si != 0:
                    if inc < np.pi / 2:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[: imax + 1] = np.nan
                    else:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[imax:] = np.nan

            # Rotate by the obliquity
            xr = -x * co - y * so
            yr = -x * so + y * co
            res.append((xr, yr))

    return res


def get_projection(projection):
    """

    """
    if projection.lower().startswith("rect"):
        projection = STARRY_RECTANGULAR_PROJECTION
    elif projection.lower().startswith("ortho"):
        projection = STARRY_ORTHOGRAPHIC_PROJECTION
    elif projection.lower().startswith("moll"):
        projection = STARRY_MOLLWEIDE_PROJECTION
    else:
        raise ValueError("Unknown map projection.")
    return projection
