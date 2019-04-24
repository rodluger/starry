# -*- coding: utf-8 -*-
"""Healpy-based spherical harmonic transform utilities for starry."""
import numpy as np
try:
    import healpy as hp
except ImportError:
    hp = None
from PIL import Image
from matplotlib.image import pil_to_array
import os


__all__ = ["image2map", "healpix2map", "array2map"]


def healpix2map(healpix_map, lmax=10, **kwargs):
    """Return a map vector corresponding to a healpix array."""
    if hp is None:
        raise ImportError("Please install the `healpy` Python package to " +
                          "enable this feature. See " +
                          "`https://healpy.readthedocs.io`.")
    # Get the complex spherical harmonic coefficients
    alm = hp.sphtfunc.map2alm(healpix_map, lmax=lmax)
    
    # We first need to do a rotation to get our axes aligned correctly,
    # since we use a different convention than `healpy`
    alm = hp.rotator.Rotator((-90, 0, -90)).rotate_alm(alm)

    # Smooth the map?
    if kwargs.get("sigma", None) is not None:
        alm = hp.sphtfunc.smoothalm(alm, sigma=kwargs.get("sigma"), verbose=False)

    # Convert them to real coefficients
    ylm = np.zeros(lmax ** 2 + 2 * lmax + 1, dtype='float')
    i = 0
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            j = hp.sphtfunc.Alm.getidx(lmax, l, np.abs(m))
            if m < 0:
                ylm[i] = np.sqrt(2) * (-1) ** m * alm[j].imag
            elif m == 0:
                ylm[i] = alm[j].real
            else:
                ylm[i] = np.sqrt(2) * (-1) ** m * alm[j].real
            i += 1

    # Normalize and return
    ylm /= ylm[0]

    return ylm


def image2map(image, lmax=10, **kwargs):
    """Return a map vector corresponding to a lat-long map image."""
    # If image doesn't exist, check for it in `img` directory
    if not os.path.exists(image):
        dn = os.path.dirname
        image = os.path.join(dn(dn(os.path.abspath(__file__))), "img", image)
        if not image.endswith(".jpg"):
            image += ".jpg"
        if not os.path.exists(image):
            raise ValueError("File not found: %s." % image)

    # Get the image array
    grayscale_pil_image = Image.open(image).convert("L")
    image_array = pil_to_array(grayscale_pil_image)
    image_array = np.array(image_array, dtype=float)
    image_array /= np.max(image_array)

    # Convert it to a map
    return array2map(image_array, lmax=lmax, **kwargs)


def array2map(image_array, lmax=10, sampling_factor=8, **kwargs):
    """Return a map vector corresponding to a lat-lon map image array."""
    if hp is None:
        raise ImportError("Please install the `healpy` Python package to " +
                          "enable this feature. See " +
                          "`https://healpy.readthedocs.io`.")
    # Figure out a reasonable number of sides
    # TODO: This is not optimized. There may be a better criterion
    # for figuring out the optimal number of sides.
    npix = image_array.shape[0] * image_array.shape[1]
    nside = 2
    while hp.nside2npix(nside) * sampling_factor < npix:
        nside *= 2

    # Convert to a healpix map
    theta = np.linspace(0, np.pi, image_array.shape[0])[:, None]
    phi = np.linspace(-np.pi, np.pi, image_array.shape[1])[::-1]
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    healpix_map = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    healpix_map[pix] = image_array

    # Now convert it to a spherical harmonic map
    return healpix2map(healpix_map, lmax=lmax, **kwargs)