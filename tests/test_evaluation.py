"""Test the map evaluation."""
import starry2
import numpy as np


def run(temporal=False, spectral=False, multi=False):
    """Compare the map evaluation to some benchmarks."""
    # Instantiate
    lmax = 2
    if spectral:
        nw = 2
        map = starry2.Map(lmax, multi=multi, nw=nw)
    elif temporal:
        nw = 1
        map = starry2.Map(lmax, multi=multi, nt=2)
    else:
        nw = 1
        map = starry2.Map(lmax, multi=multi)
    map.axis = [0, 1, 0]
    map[:, :] = 1

    # No arguments
    I = map()
    assert np.allclose(I, 1.5814013250227599)

    # Scalar evaluation
    I = map(x=0.1, y=0.1)
    assert np.allclose(I, 1.9211848890432843)

    # Vector evaluation
    I = map(x=[0.1, 0.2, 0.3], y=[0.1, 0.2, 0.3])
    truth = np.repeat([1.9211848890432843, 
                       2.216308435590377, 
                       2.44870978566566], nw).reshape(3, -1)
    assert np.allclose(I, np.squeeze(truth))


def test_evaluation_single_double():
    """Test the map evaluation against some benchmarks [single, double]."""
    return run(temporal=False, spectral=False, multi=False)


def test_evaluation_single_multi():
    """Test the map evaluation against some benchmarks [single, multi]."""
    return run(temporal=False, spectral=False, multi=True)


def test_evaluation_spectral_double():
    """Test the map evaluation against some benchmarks [spectral, double]."""
    return run(temporal=False, spectral=True, multi=False)


def test_evaluation_spectral_multi():
    """Test the map evaluation against some benchmarks [spectral, multi]."""
    return run(temporal=False, spectral=True, multi=True)


def test_evaluation_temporal_double():
    """Test the map evaluation against some benchmarks [temporal, double]."""
    return run(temporal=True, spectral=False, multi=False)


def test_evaluation_temporal_multi():
    """Test the map evaluation against some benchmarks [temporal, multi]."""
    return run(temporal=True, spectral=False, multi=True)