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
import pymc3 as pm


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


def test_show_ld():
    map = starry.Map(udeg=2)
    map.show(file="tmp.pdf")
    os.remove("tmp.pdf")


def test_system_show():
    pri = starry.Primary(starry.Map())
    sec = starry.Secondary(starry.Map(), porb=1.0)
    sys = starry.System(pri, sec)
    sys.show(0.1, file="tmp.pdf")
    os.remove("tmp.pdf")
    sys.show([0.1, 0.2], file="tmp.mp4")
    os.remove("tmp.mp4")
    sys.show([0.1, 0.2], file="tmp.gif")
    os.remove("tmp.gif")


def test_system_rv_show():
    pri = starry.Primary(starry.Map(rv=True))
    sec = starry.Secondary(starry.Map(rv=True), porb=1.0)
    sys = starry.System(pri, sec)
    sys.show(0.1, file="tmp.pdf")
    os.remove("tmp.pdf")
    sys.show([0.1, 0.2], file="tmp.mp4")
    os.remove("tmp.mp4")


def test_show_pymc3():
    with pm.Model(theano_config=dict(compute_test_value="ignore")) as model:
        map = starry.Map()
        ncoeff = map.Ny
        mu = np.ones(ncoeff)
        cov = 1e-2 * np.eye(ncoeff)
        map[:, :] = pm.MvNormal("y", mu, cov, shape=(ncoeff,))
        map.show(theta=0.1, file="tmp.pdf", point=model.test_point)
        os.remove("tmp.pdf")


def test_show_reflected_pymc3():
    with pm.Model(theano_config=dict(compute_test_value="ignore")) as model:
        map = starry.Map(ydeg=1, udeg=1, reflected=True)
        ncoeff = map.Ny
        mu = np.ones(ncoeff)
        cov = 1e-2 * np.eye(ncoeff)
        map[:, :] = pm.MvNormal("y", mu, cov, shape=(ncoeff,))
        map.show(theta=0.1, file="tmp.pdf", point=model.test_point)
        os.remove("tmp.pdf")


def test_show_rv_pymc3():
    with pm.Model(theano_config=dict(compute_test_value="ignore")) as model:
        map = starry.Map(ydeg=1, udeg=1, rv=True)
        ncoeff = map.Ny
        mu = np.ones(ncoeff)
        cov = 1e-2 * np.eye(ncoeff)
        map[:, :] = pm.MvNormal("y", mu, cov, shape=(ncoeff,))
        map.show(rv=True, theta=0.1, file="tmp.pdf", point=model.test_point)
        os.remove("tmp.pdf")


def test_show_ld_pymc3():
    with pm.Model(theano_config=dict(compute_test_value="ignore")) as model:
        map = starry.Map(udeg=2)
        map[1:] = pm.MvNormal("u", [0.5, 0.25], np.eye(2), shape=(2,))
        map.show(file="tmp.pdf", point=model.test_point)
        os.remove("tmp.pdf")


def test_system_show_pymc3():
    with pm.Model(theano_config=dict(compute_test_value="ignore")) as model:
        pri = starry.Primary(starry.Map())
        sec = starry.Secondary(starry.Map(), porb=1.0)
        sec.inc = pm.Uniform("inc", 0, 90)
        sys = starry.System(pri, sec)
        sys.show(0.1, file="tmp.pdf", point=model.test_point)
        os.remove("tmp.pdf")
        sys.show([0.1, 0.2], file="tmp.gif", point=model.test_point)
        os.remove("tmp.gif")


def test_system_rv_show_pymc3():
    with pm.Model(theano_config=dict(compute_test_value="ignore")) as model:
        pri = starry.Primary(starry.Map(rv=True))
        sec = starry.Secondary(starry.Map(rv=True), porb=1.0)
        sec.inc = pm.Uniform("inc", 0, 90)
        sys = starry.System(pri, sec)
        sys.show(0.1, file="tmp.pdf", point=model.test_point)
        os.remove("tmp.pdf")
