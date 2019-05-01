# -*- coding: utf-8 -*-
from inspect import getmro
import theano
import theano.tensor as tt
import numpy as np
from ..extensions import RAxisAngle


__all__ = ["is_theano",
           "to_tensor",
           "vectorize", 
           "get_ortho_latitude_lines", 
           "get_ortho_longitude_lines"]


def is_theano(*objs):
    """

    """
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


def to_tensor(*args):
    """

    """
    return [tt.as_tensor_variable(arg).astype(tt.config.floatX) for arg in args]


def vectorize(*args):
    """

    """
    if is_theano(*args):
        ones = tt.ones_like(np.sum([arg if is_theano(arg) 
                                    else np.atleast_1d(arg) 
                                    for arg in args], axis=0)).astype(tt.config.floatX).reshape([-1])
        args = tuple([arg.astype(tt.config.floatX) * ones for arg in args])
    else:
        ones = np.ones_like(np.sum([np.atleast_1d(arg) for arg in args], axis=0))
        args = tuple([arg * ones for arg in args])
    return args


def get_ortho_latitude_lines(inc=90, obl=0, nlines=5, npts=1000):
    """

    """
    # Angular quantities
    ci = np.cos(inc * np.pi / 180)
    si = np.sin(inc * np.pi / 180)
    co = np.cos(obl * np.pi / 180)
    so = np.sin(obl * np.pi / 180)

    # Latitude lines
    res = []
    latlines = np.linspace(-90, 90, nlines + 2)[1:-1]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat * np.pi / 180) * si
        a = np.cos(lat * np.pi / 180)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if (si != 0):
            if inc > 90:
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
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))

    return res


def get_ortho_longitude_lines(inc=90, obl=0, nlines=13, npts=1000):
    """

    """
    # Angular quantities
    ci = np.cos(inc * np.pi / 180)
    si = np.sin(inc * np.pi / 180)
    co = np.cos(obl * np.pi / 180)
    so = np.sin(obl * np.pi / 180)

    # Longitude lines
    res = []
    lonlines = np.linspace(-180, 180, nlines)
    for lon in lonlines:
        # Viewed at i = 90
        b = np.sin(lon * np.pi / 180)
        y = np.linspace(-1, 1, npts)
        x = b * np.sqrt(1 - y ** 2)
        z = np.sqrt(np.abs(1 - x ** 2 - y ** 2))

        if (inc > 88) and (inc < 92):
            y1 = y
            y2 = np.nan * y
        else:
            # Rotate by the inclination
            R = RAxisAngle([1, 0, 0], 90 - inc)
            v = np.vstack((x.reshape(1, -1), 
                            y.reshape(1, -1), 
                            z.reshape(1, -1)))
            x, y1, _ = np.dot(R, v)
            v[2] *= -1
            _, y2, _ = np.dot(R, v)

            # Mask lines on the backside
            if (si != 0):
                if inc < 90:
                    imax = np.argmax(x ** 2 + y1 ** 2)
                    y1[:imax + 1] = np.nan
                    imax = np.argmax(x ** 2 + y2 ** 2)
                    y2[:imax + 1] = np.nan
                else:
                    imax = np.argmax(x ** 2 + y1 ** 2)
                    y1[imax:] = np.nan
                    imax = np.argmax(x ** 2 + y2 ** 2)
                    y2[imax:] = np.nan

        # Rotate them
        for y in (y1, y2):
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))
    
    return res