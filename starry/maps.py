"""Map fitting utilities for starry."""
import numpy as np
import healpy as hp
from PIL import Image
from matplotlib.image import pil_to_array
import os


__all__ = ["image2map", "healpix2map"]


def healpix2map(healpix_map, lmax=10):
    """Return a map vector corresponding to a healpix array."""
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
        image = os.path.join(dn(os.path.abspath(__file__)),
                             "maps", image + ".jpg")
        if not os.path.exists(image):
            raise ValueError("File not found: %s." % image)

    # Get the image array
    grayscale_pil_image = Image.open(image).convert("L")
    image_array = pil_to_array(grayscale_pil_image)

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
