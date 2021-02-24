# -*- coding: utf-8 -*-
"""
Functions to test map importing / SH transforms

"""
import starry
import numpy as np
from PIL import Image
from matplotlib.image import pil_to_array
import os
import pytest
from skimage.feature import canny
from scipy.ndimage import shift


def test_load():
    """Test that we're loading images correctly."""
    # Get the image directly
    image = os.path.join(os.path.dirname(starry.__file__), "img", "spot.png")
    grayscale_pil_image = Image.open(image).convert("L")
    img1 = pil_to_array(grayscale_pil_image)
    img1 = np.array(img1, dtype=float)
    img1 /= 255.0  # to [0, 1] range
    img1 = np.flipud(img1)  # align
    img1 = img1[:, ::2]  # make square to match starry output

    # Render the image with starry at high res
    map = starry.Map(30)
    map.load("spot", extent=(-180, 179, -90, 89))
    img2 = map.render(projection="rect", res=img1.shape[0])

    # Canny filter on both images to detect edges
    img1 = np.array(canny(img1, sigma=4), dtype=int)
    img2 = np.array(canny(img2, sigma=4), dtype=int)

    # Compute the difference between the images
    # Do this several times by translating one of
    # the images by a single pixel and computing
    # the *minimum*. This has the effect of allowing
    # a one-pixel tolerance in the comparison
    diff = np.min(
        np.abs(
            [
                img2 - img1,
                img2 - shift(img1, [0, 1]),
                img2 - shift(img1, [0, -1]),
                img2 - shift(img1, [1, 0]),
                img2 - shift(img1, [-1, 0]),
            ]
        ),
        axis=0,
    )

    # Number of pixels in the image edges
    pixels = np.count_nonzero(img1)

    # Number of pixels that differ between the two
    pixels_diff = np.count_nonzero(diff)

    # There should be VERY few pixels in the difference image
    assert pixels_diff / pixels < 0.01


def test_load_normalization():
    """Test that we're normalizing imported images correctly."""
    # Render a band-limited image
    map = starry.Map(5)
    map[5, 3] = 1
    img1 = map.render(projection="moll")
    img1_rect = map.render(projection="rect")

    # Now load that image and re-render it;
    # the resulting map should be *very*
    # close to the original map!
    map.reset()
    map.load(img1_rect, fac=np.inf, smoothing=0)
    img2 = map.render(projection="moll")

    # Compute stats
    mu1 = np.nanmean(img1)
    sig1 = np.nanstd(img1)
    mu2 = np.nanmean(img2)
    sig2 = np.nanstd(img2)

    # Assert diff is small
    assert np.abs(mu1 - mu2) < 1e-3
    assert np.abs(sig1 - sig2) < 1e-3


def test_force_psd():
    """Test the positive semi-definite feature."""
    # High res map
    map = starry.Map(10)

    # Render the image with starry with aggressive minimum finding
    map.load("spot", force_psd=True, oversample=3, ntries=2)

    # Ensure positive everywhere
    assert map.render(projection="rect").min() >= 0
