# -*- coding: utf-8 -*-
"""
Functions to test map importing / SH transforms

"""
import starry
import numpy as np
from PIL import Image
from matplotlib.image import pil_to_array
import os


def test_load_normalization():
    """Test that we're normalizing imported images correctly."""
    # High res map
    map = starry.Map(30)

    # Render the image with starry
    map.load("jupiter")
    img1 = map.render(projection="rect", res=360)
    img1 = img1.flatten()

    # Get the image directly
    earth = os.path.join(
        os.path.dirname(starry.__file__), "img", "jupiter.jpg"
    )
    grayscale_pil_image = Image.open(earth).convert("L")
    img2 = pil_to_array(grayscale_pil_image)
    img2 = np.array(img2, dtype=float)
    img2 /= 255.0  # to [0, 1] range
    img2 = np.flipud(img2)  # align
    img2 = img2[::2, ::4]  # downsample
    img2 = img2.flatten()

    # Compute stats
    mu1 = np.mean(img1)
    sig1 = np.std(img1)
    mu2 = np.mean(img2)
    sig2 = np.std(img2)
    print(sig2 - sig1)

    # Assert within a couple percent
    assert np.abs(mu1 - mu2) < 0.01
    assert np.abs(sig1 - sig2) < 0.02


def test_force_psd():
    """Test the positive semi-definite feature."""
    # High res map
    map = starry.Map(10)

    # Render the image with starry with aggressive minimum finding
    map.load("spot", force_psd=True, oversample=3, ntries=2)

    # Ensure positive everywhere
    assert map.render(projection="rect").min() >= 0
