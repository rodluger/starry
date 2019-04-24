"""Test flux and intensity output dimensions."""
import numpy as np
import starry
import pytest


# Settings
nw = 2
nt = 2
ydeg = 3
udeg = 2
npts = 100

# Let's only compute phase curves for simplicity
t = 0.0
theta = np.linspace(0.0, 360.0, npts)
xo = 10.0
yo = 10.0
ro = 0.1


def test_default_no_grad():
    map = starry.Map(ydeg=ydeg, udeg=0)
    flux = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (npts,))
    intensity = map(theta=theta, x=xo, y=yo)
    assert(intensity.shape == (npts,))


def test_limbdarkened():
    map = starry.Map(ydeg=0, udeg=udeg)
    map[1:] = 1
    flux, grad = map.flux(b=np.linspace(-1, 1, npts), ro=ro, gradient=True)       
    assert(flux.shape == grad['b'].shape == grad['ro'].shape == (npts,))
    assert(grad['u'].shape == (udeg, npts))


def test_default_ylm():
    map = starry.Map(ydeg=ydeg, udeg=0)
    map[:, :] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((ydeg + 1) ** 2 - 1, npts))
    assert(len(grad['u']) == 0)


def test_default_ylm_ld():
    map = starry.Map(ydeg=ydeg, udeg=udeg)
    map[:, :] = 1
    map[1:] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((ydeg + 1) ** 2 - 1, npts))
    assert(grad['u'].shape == (udeg, npts))


def test_spectral_no_grad():
    map = starry.Map(ydeg=ydeg, udeg=0, nw=nw)
    flux = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (npts, nw))
    intensity = map(theta=theta, x=xo, y=yo)
    assert(intensity.shape == (npts, nw))


def test_spectral_ylm():
    map = starry.Map(ydeg=ydeg, udeg=0, nw=nw)
    map[:, :, :] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == ((ydeg + 1) ** 2 - 1, npts, nw))
    assert(len(grad['u']) == 0)


def test_spectral_ylm_ld():
    map = starry.Map(ydeg=ydeg, udeg=udeg, nw=nw)
    map[:, :, :] = 1
    map[1:] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == ((ydeg + 1) ** 2 - 1, npts, nw))
    assert(grad['u'].shape == (udeg, npts, nw))


def test_temporal_no_grad():
    map = starry.Map(ydeg=ydeg, udeg=0, nt=nt)
    map[1:, :, :] = 1
    flux = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (npts,))
    intensity = map(t=t, theta=theta, x=xo, y=yo)
    assert(intensity.shape == (npts,))


def test_temporal_ylm():
    map = starry.Map(ydeg=ydeg, udeg=0, nt=nt)
    map[1:, :, :] = 1
    flux, grad = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == (nt * ((ydeg + 1) ** 2 - 1), npts))
    assert(len(grad['u']) == 0)


def test_temporal_ylm_ld():
    map = starry.Map(ydeg=ydeg, udeg=udeg, nt=nt)
    map[1:, :, :] = 1
    map[1:] = 1
    flux, grad = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == (nt * ((ydeg + 1) ** 2 - 1), npts))
    assert(grad['u'].shape == (udeg, npts))