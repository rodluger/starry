"""Test flux and intensity output dimensions."""
import numpy as np
import starry2


def test_shapes():
    """Test that all output has the correct shapes."""
    # Settings
    nw = 2
    nt = 2
    lmax = 3
    npts = 100
    t = 0.0
    theta = 10.0
    xo = np.linspace(-1.5, 1.5, npts)
    yo = 0.0
    ro = 0.1

    # Single wavelength
    map = starry2.Map(lmax)
    flux = map.flux(theta, xo, yo, ro, False)
    assert(flux.shape == (npts,))
    intensity = map(theta, xo, yo)
    assert(intensity.shape == (npts,))

    # Single wavelength, limb darkened (with gradient)
    map = starry2.Map(lmax)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == (1, npts))
    assert(grad['u'].shape == (lmax, npts))

    # Single wavelength, Ylm (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax)
    map[:, :] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts))
    assert(len(grad['u']) == 0)
    '''

    # Single wavelength, Ylm + LD (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax)
    map[:lmax, :]
    map[1] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts))
    assert(grad['u'].shape == (lmax, npts))
    '''

    # Spectral
    map = starry2.Map(lmax, nw=nw)
    flux = map.flux(theta, xo, yo, ro, False)
    assert(flux.shape == (npts, nw))
    intensity = map(theta, xo, yo)
    assert(intensity.shape == (npts, nw))

    # Spectral, limb darkened (with gradient)
    map = starry2.Map(lmax, nw=nw)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == (1, npts, nw))
    assert(grad['u'].shape == (lmax, npts, nw))

    # Spectral, Ylm (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nw=nw)
    map[:, :] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts, nw))
    assert(len(grad['u']) == 0)
    '''

    # Spectral, Ylm + LD (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nw=nw)
    map[:lmax, :]
    map[1] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (npts, nw))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts, nw))
    assert(grad['u'].shape == (lmax, npts, nw))
    '''

    # Temporal
    map = starry2.Map(lmax, nt=nt)
    flux = map.flux(theta, xo, yo, ro, False)
    assert(flux.shape == (npts,))
    intensity = map(theta, xo, yo)
    assert(intensity.shape == (npts,))

    # Temporal, limb darkened (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nt=nt)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(t, theta, xo, yo, ro, True)    
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == (1, npts))
    assert(grad['u'].shape == (lmax, npts))
    '''

    # Temporal, Ylm (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax)
    map[:, :] = 1
    flux, grad = map.flux(t, theta, xo, yo, ro, True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts))
    assert(len(grad['u']) == 0)
    '''

    # Temporal, Ylm + LD (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax)
    map[:lmax, :]
    map[1] = 1
    flux, grad = map.flux(t, theta, xo, yo, ro, True)       
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
        grad['yo'].shape == grad['ro'].shape == (npts,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, npts))
    assert(grad['u'].shape == (lmax, npts))
    '''

def test_shapes_single_cadence():
    """Test that all output has the correct shapes."""
    # Settings
    nw = 2
    nt = 2
    lmax = 3
    theta = 10.0
    t = 0.0
    xo = 0.0
    yo = 0.0
    ro = 0.1

    # Single wavelength
    map = starry2.Map(lmax)
    flux = map.flux(theta, xo, yo, ro, False)
    assert(type(flux) is float)
    intensity = map(theta, xo, yo)
    assert(type(intensity) is float)

    # Single wavelength, limb darkened (with gradient)
    map = starry2.Map(lmax)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == (1,))
    assert(grad['u'].shape == (lmax,))

    # Single wavelength, Ylm (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax)
    map[:, :] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2,))
    assert(len(grad['u']) == 0)
    '''

    # Single wavelength, Ylm + LD (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax)
    map[:lmax, :]
    map[1] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2,))
    assert(grad['u'].shape == (lmax,))
    '''

    # Spectral
    map = starry2.Map(lmax, nw=nw)
    flux = map.flux(theta, xo, yo, ro, False)
    assert(flux.shape == (nw,))
    intensity = map(theta, xo, yo)
    assert(intensity.shape == (nw,))

    # Spectral, limb darkened (with gradient)
    map = starry2.Map(lmax, nw=nw)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (nw,))
    assert(grad['y'].shape == (1, nw))
    assert(grad['u'].shape == (lmax, nw))

    # Spectral, Ylm (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nw=nw)
    map[:, :] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (nw,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, nw))
    assert(len(grad['u']) == 0)
    '''

    # Spectral, Ylm + LD (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nw=nw)
    map[:lmax, :]
    map[1] = 1
    flux, grad = map.flux(theta, xo, yo, ro, True)
    assert(flux.shape == grad['theta'].shape == grad['xo'].shape == 
           grad['yo'].shape == grad['ro'].shape == (nw,))
    assert(grad['y'].shape == ((lmax + 1) ** 2, nw))
    assert(grad['u'].shape == (lmax, nw))
    '''

    # Temporal
    map = starry2.Map(lmax, nt=nt)
    flux = map.flux(t, theta, xo, yo, ro, False)
    assert(type(flux) is float)
    intensity = map(theta, xo, yo)
    assert(type(intensity) is float)

    # Temporal, limb darkened (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nt=nt)
    map[0, 0] = 1
    map[:] = 1
    flux, grad = map.flux(t, theta, xo, yo, ro, True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == (1,))
    assert(grad['u'].shape == (lmax,))
    '''

    # Temporal, Ylm (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nt=nt)
    map[:, :] = 1
    flux, grad = map.flux(t, theta, xo, yo, ro, True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2,))
    assert(len(grad['u']) == 0)
    '''

    # Temporal, Ylm + LD (with gradient)
    # TODO: Not yet implemented on the C++ side
    '''
    map = starry2.Map(lmax, nt=nt)
    map[:lmax, :]
    map[1] = 1
    flux, grad = map.flux(t, theta, xo, yo, ro, True)       
    assert(type(flux) == type(grad['theta']) == type(grad['xo']) == 
           type(grad['yo']) == type(grad['ro']) == float)
    assert(grad['y'].shape == ((lmax + 1) ** 2,))
    assert(grad['u'].shape == (lmax,))
    '''


if __name__ == "__main__":
    test_shapes()
    test_shapes_single_cadence()