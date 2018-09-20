"""Map fitting utilities for starry."""
import numpy as np
try:
    import healpy as hp
except ImportError:
    hp = None
from PIL import Image
from matplotlib.image import pil_to_array
from scipy.special import ive as BesselI
import os


__all__ = ["load_map", "image2map", "healpix2map"]


def load_map(image, lmax=10, healpix=False):
    """Allow the user to specify an image, array, or healpix map."""
    if type(image) is str:
        y = image2map(image, lmax=lmax)
    # Or is this an array?
    elif (type(image) is np.ndarray):
        if healpix:
            y = healpix2map(image, lmax=lmax)
        else:
            y = array2map(image, lmax=lmax)
    else:
        raise ValueError("Invalid `image` value.")
    return y


def healpix2map(healpix_map, lmax=10):
    """Return a map vector corresponding to a healpix array."""
    if hp is None:
        raise ImportError("Please install the `healpy` Python package to " +
                          "enable this feature. See " +
                          "`https://healpy.readthedocs.io`.")
    # Get the complex spherical harmonic coefficients
    alm = hp.sphtfunc.map2alm(healpix_map, lmax=lmax)

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

    return ylm


def image2map(image, lmax=10):
    """Return a map vector corresponding to a lat-long map image."""
    # If image doesn't exist, check for it in maps directory
    if not os.path.exists(image):
        dn = os.path.dirname
        image = os.path.join(dn(os.path.abspath(__file__)), image + ".jpg")
        if not os.path.exists(image):
            raise ValueError("File not found: %s." % image)

    # Get the image array
    grayscale_pil_image = Image.open(image).convert("L")
    image_array = pil_to_array(grayscale_pil_image)
    image_array = np.array(image_array, dtype=float)
    image_array /= np.max(image_array)

    # Convert it to a map
    return array2map(image_array, lmax=lmax)


def array2map(image_array, lmax=10):
    """Return a map vector corresponding to a lat-lon map image array."""
    if hp is None:
        raise ImportError("Please install the `healpy` Python package to " +
                          "enable this feature. See " +
                          "`https://healpy.readthedocs.io`.")
    # Figure out a reasonable number of sides
    # TODO: Not optimized!
    npix = image_array.shape[0] * image_array.shape[1]
    nside = 2
    while hp.nside2npix(nside) * 8 < npix:
        nside *= 2

    # Convert to a healpix map
    theta = np.linspace(0, np.pi, image_array.shape[0])[:, None]
    phi = np.linspace(-np.pi, np.pi, image_array.shape[1])[::-1]
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    healpix_map = np.zeros(hp.nside2npix(nside), dtype=np.double)
    healpix_map[pix] = image_array

    # Now convert it to a spherical harmonic map
    return healpix2map(healpix_map, lmax=lmax)


def gaussian(sigma=0.1, lmax=10, res=500):
    """Return a spherical harmonic expansion of a Gaussian."""
    lon = np.linspace(-np.pi, np.pi, res * 2)
    lat = np.linspace(-np.pi / 2, np.pi / 2, res)
    lon, lat = np.meshgrid(lon, lat)
    z = np.cos(lat) * np.cos(lon)
    w = sigma ** -2
    norm = np.pi * BesselI(0, w)
    g = norm * np.exp((z - 1) / sigma ** 2)
    y = array2map(g, lmax=lmax)
    # NOTE: Force the constant term to zero so we
    # add no net flux. We need to think carefully
    # about this.
    y[0] = 0
    return y
