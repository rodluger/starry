# -*- coding: utf-8 -*-
"""Healpy-based spherical harmonic transform utilities for starry."""
import numpy as np
from PIL import Image
from matplotlib.image import pil_to_array
import os
from scipy import ndimage

try:
    import healpy as hp
except ImportError:
    hp = None

__all__ = ["image2map", "healpix2map", "array2map", "array2healpix"]


def healpix2map(healpix_map, lmax=10, **kwargs):
    """Return a map vector corresponding to a healpix array."""
    if hp is None:
        raise ImportError(
            "Please install the `healpy` Python package to "
            "enable this feature. See `https://healpy.readthedocs.io`."
        )
    # Get the complex spherical harmonic coefficients
    alm = hp.sphtfunc.map2alm(healpix_map, lmax=lmax)

    # We first need to do a rotation to get our axes aligned correctly,
    # since we use a different convention than `healpy`
    alm = hp.rotator.Rotator((-90, 0, -90)).rotate_alm(alm)

    # Smooth the map?
    sigma = kwargs.pop("sigma", None)
    if sigma is not None:
        alm = hp.sphtfunc.smoothalm(alm, sigma=sigma, verbose=False)

    # Convert them to real coefficients
    ylm = np.zeros(lmax ** 2 + 2 * lmax + 1, dtype="float")
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

    return ylm


def image2map(image, **kwargs):
    """Return a map vector corresponding to a lat-long map image."""
    # If image doesn't exist, check for it in `img` directory
    if not os.path.exists(image):
        dn = os.path.dirname
        image = os.path.join(dn(os.path.abspath(__file__)), "img", image)
        if not image.endswith(".jpg"):
            image += ".jpg"
        if not os.path.exists(image):
            raise ValueError("File not found: %s." % image)

    # Get the image array
    grayscale_pil_image = Image.open(image).convert("L")
    image_array = pil_to_array(grayscale_pil_image)
    image_array = np.array(image_array, dtype=float)
    image_array /= 255.0

    # Convert it to a map
    return array2map(image_array, **kwargs)


def array2healpix(image_array, nside=16, max_iter=3, **kwargs):
    """Return a healpix ring-ordered map corresponding to a lat-lon map image array."""
    if hp is None:
        raise ImportError(
            "Please install the `healpy` Python package to "
            "enable this feature. See `https://healpy.readthedocs.io`."
        )

    # Starting value for the zoom
    zoom = 2

    # Keep track of the number of unseen pixels
    unseen = 1
    ntries = 0
    while unseen > 0:

        # Make the image bigger so we have good angular coverage
        image_array = ndimage.zoom(image_array, zoom)

        # Convert to a healpix map
        theta = np.linspace(0, np.pi, image_array.shape[0])[:, None]
        phi = np.linspace(-np.pi, np.pi, image_array.shape[1])[::-1]
        pix = hp.ang2pix(nside, theta, phi, nest=False)
        healpix_map = (
            np.ones(hp.nside2npix(nside), dtype=np.float64) * hp.UNSEEN
        )
        healpix_map[pix] = image_array

        # Count the unseen pixels
        unseen = np.count_nonzero(healpix_map == hp.UNSEEN)

        # Did we do this too many times?
        ntries += 1
        if ntries > max_iter:
            raise ValueError(
                "Maximum number of iterations exceeded. Either decreaser `nside` or increase `max_iter`."
            )

    return healpix_map


def array2map(image_array, **kwargs):
    """Return a map vector corresponding to a lat-lon map image array."""
    # Get the healpix map
    healpix_map = array2healpix(image_array, **kwargs)

    # Now convert it to a spherical harmonic map
    return healpix2map(healpix_map, **kwargs)
