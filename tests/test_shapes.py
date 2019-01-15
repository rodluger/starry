"""Test flux and intensity output dimensions."""
import numpy as np
import starry2

# Settings
nw = 2
nt = 2
lmax = 3
npts = 100

# Let's only compute phase curves for simplicity
t = 0.0
theta = np.linspace(0.0, 360.0, npts)
xo = 10.0
yo = 10.0
ro = 0.1


def test_default_no_grad():
    map = starry2.Map(lmax)
    flux = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (npts,))
    intensity = map(theta=theta, x=xo, y=yo)
    assert(intensity.shape == (npts,))


def test_default_ld():
    map = starry2.Map(lmax)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == (1, npts))
    assert(grad['u'].shape == (lmax, npts))


def test_default_ylm():
    map = starry2.Map(lmax)
    map[:, :] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts))
    assert(len(grad['u']) == 0)


def test_default_ylm_ld():
    map = starry2.Map(lmax)
    map[:lmax, :] = 1
    map[1] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts))
    assert(grad['u'].shape == (lmax, npts))


def test_spectral_no_grad():
    map = starry2.Map(lmax, nw=nw)
    flux = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (npts, nw))
    intensity = map(theta=theta, x=xo, y=yo)
    assert(intensity.shape == (npts, nw))


def test_spectral_ld():
    map = starry2.Map(lmax, nw=nw)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == (1, npts, nw))
    assert(grad['u'].shape == (lmax, npts, nw))


def test_spectral_ylm():
    map = starry2.Map(lmax, nw=nw)
    map[:, :] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts, nw))
    assert(len(grad['u']) == 0)


def test_spectral_ylm_ld():
    map = starry2.Map(lmax, nw=nw)
    map[:lmax, :] = 1
    map[1] = 1
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts, nw))
    assert(grad['u'].shape == (lmax, npts, nw))


def test_temporal_no_grad():
    map = starry2.Map(lmax, nt=nt)
    flux = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (npts,))
    intensity = map(t=t, theta=theta, x=xo, y=yo)
    assert(intensity.shape == (npts,))


def test_temporal_ld():
    map = starry2.Map(lmax, nt=nt)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)    
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == (1, npts, nt))
    assert(grad['u'].shape == (lmax, npts))


def test_temporal_ylm():
    map = starry2.Map(lmax, nt=nt)
    map[:, :] = 1
    flux, grad = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts, nt))
    assert(len(grad['u']) == 0)


def test_temporal_ylm_ld():
    map = starry2.Map(lmax, nt=nt)
    map[:lmax, :] = 1
    map[1] = 1
    flux, grad = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts, nt))
    assert(grad['u'].shape == (lmax, npts))


def test_default_no_grad_single():
    map = starry2.Map(lmax)
    flux = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=False)
    assert(type(flux) is float)
    intensity = map(theta=theta[0], x=xo, y=yo)
    assert(type(intensity) is float)


def test_default_ld_single():
    map = starry2.Map(lmax)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == (1,))
    assert(grad['u'].shape == (lmax,))


def test_default_ylm_single():
    map = starry2.Map(lmax)
    map[:, :] = 1
    flux, grad = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2,))
    assert(len(grad['u']) == 0)


def test_default_ylm_ld_single():
    map = starry2.Map(lmax)
    map[:lmax, :] = 1
    map[1] = 1
    flux, grad = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2,))
    assert(grad['u'].shape == (lmax,))


def test_spectral_no_grad_single():
    map = starry2.Map(lmax, nw=nw)
    flux = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=False)
    assert(flux.shape == (nw,))
    intensity = map(theta=theta[0], x=xo, y=yo)
    assert(intensity.shape == (nw,))


def test_spectral_ld_single():
    map = starry2.Map(lmax, nw=nw)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (nw,))
    assert(grad['y'].shape == (1, nw))
    assert(grad['u'].shape == (lmax, nw))


def test_spectral_ylm_single():
    map = starry2.Map(lmax, nw=nw)
    map[:, :] = 1
    flux, grad = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (nw,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, nw))
    assert(len(grad['u']) == 0)


def test_spectral_ylm_ld_single():
    map = starry2.Map(lmax, nw=nw)
    map[:lmax, :] = 1
    map[1] = 1
    flux, grad = map.flux(theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (nw,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, nw))
    assert(grad['u'].shape == (lmax, nw))


def test_temporal_no_grad_single():
    map = starry2.Map(lmax, nt=nt)
    flux = map.flux(t=t, theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=False)
    assert(type(flux) is float)
    intensity = map(theta=theta[0], x=xo, y=yo)
    assert(type(intensity) is float)


def test_temporal_ld_single():
    map = starry2.Map(lmax, nt=nt)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(t=t, theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)      
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == (1, nt))
    assert(grad['u'].shape == (lmax,))


def test_temporal_ylm_single():
    map = starry2.Map(lmax, nt=nt)
    map[:, :] = 1
    flux, grad = map.flux(t=t, theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)         
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2, nt))
    assert(len(grad['u']) == 0)


def test_temporal_ylm_ld_single():
    map = starry2.Map(lmax, nt=nt)
    map[:lmax, :] = 1
    map[1] = 1
    flux, grad = map.flux(t=t, theta=theta[0], xo=xo, yo=yo, ro=ro, gradient=True)          
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2, nt))
    assert(grad['u'].shape == (lmax,))