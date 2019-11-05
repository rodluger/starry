# -*- coding: utf-8 -*-
"""Test map visualization.

These tests don't check for anything; we're just ensuring
the code runs without raising errors.

"""
import matplotlib

matplotlib.use("Agg")

import starry
import numpy as np
import os


def test_show():
    map = starry.Map(ydeg=1, udeg=1)
    map.show(file="tmp.pdf", projection="ortho")
    os.remove("tmp.pdf")
    map.show(file="tmp.pdf", projection="rect")
    os.remove("tmp.pdf")
    map.show(theta=np.linspace(0, 360, 10), file="tmp.mp4")
    os.remove("tmp.mp4")


def test_show_reflected():
    map = starry.Map(ydeg=1, udeg=1, reflected=True)
    map.show(file="tmp.pdf", projection="ortho")
    os.remove("tmp.pdf")
    map.show(file="tmp.pdf", projection="rect")
    os.remove("tmp.pdf")
    map.show(theta=np.linspace(0, 360, 10), file="tmp.mp4")
    os.remove("tmp.mp4")


def test_show_rv():
    map = starry.Map(ydeg=1, udeg=1, rv=True)
    map.show(rv=True, file="tmp.pdf", projection="ortho")
    os.remove("tmp.pdf")
    map.show(rv=True, file="tmp.pdf", projection="rect")
    os.remove("tmp.pdf")
    map.show(rv=True, theta=np.linspace(0, 360, 10), file="tmp.mp4")
    os.remove("tmp.mp4")
