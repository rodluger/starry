# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["get_ortho_latitude_lines", 
           "get_ortho_longitude_lines"]

def RAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape([
        cost + axis[0] * axis[0] * (1 - cost),
        axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
        axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
        axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
        cost + axis[1] * axis[1] * (1 - cost),
        axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
        axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
        axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
        cost + axis[2] * axis[2] * (1 - cost)
    ], [3, 3])


def get_ortho_latitude_lines(inc=np.pi/2, obl=0, nlines=5, npts=1000):
    """

    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Latitude lines
    res = []
    latlines = np.linspace(-np.pi/2, np.pi/2, nlines + 2)[1:-1]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if (si != 0):
            if inc > np.pi/2:
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


def get_ortho_longitude_lines(inc=np.pi/2, obl=0, nlines=13, npts=1000):
    """

    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Longitude lines
    res = []
    lonlines = np.linspace(-np.pi, np.pi, nlines)
    for lon in lonlines:
        # Viewed at i = 90
        b = np.sin(lon)
        y = np.linspace(-1, 1, npts)
        x = b * np.sqrt(1 - y ** 2)
        z = np.sqrt(np.abs(1 - x ** 2 - y ** 2))

        if (inc > 88 * np.pi / 180) and (inc < 92 * np.pi / 180):
            y1 = y
            y2 = np.nan * y
        else:
            # Rotate by the inclination
            R = RAxisAngle([1, 0, 0], np.pi/2 - inc)
            v = np.vstack((x.reshape(1, -1), 
                            y.reshape(1, -1), 
                            z.reshape(1, -1)))
            x, y1, _ = np.dot(R, v)
            v[2] *= -1
            _, y2, _ = np.dot(R, v)

            # Mask lines on the backside
            if (si != 0):
                if inc < np.pi/2:
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