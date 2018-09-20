"""Test the map evaluation."""
import starry
import numpy as np
norm = 0.5 * np.sqrt(np.pi)


def run(multi=False):
    """Compare the map evaluation to some benchmarks."""
    # Instantiate
    lmax = 2
    map = starry.Map(lmax, multi=multi)
    map.axis = [0, 1, 0]
    map[:, :] = norm

    # No arguments
    I = map()
    assert np.allclose(I, 1.4014804341818383)

    # Scalar evaluation
    I = map(x=0.1, y=0.1)
    assert np.allclose(I, 1.7026057774431276)

    # Scalar evaluation
    I = map(x=0.1, y=0.1, theta=30)
    assert np.allclose(I, 0.7736072493369371)

    # Vector evaluation
    I = map(x=[0.1, 0.2, 0.3], y=[0.1, 0.2, 0.3], theta=30)
    assert np.allclose(I, [0.7736072493369371,
                           1.0432785526935853,
                           1.318434613210305])
    # Rotation caching
    I = map(x=0.1, y=0.1, theta=[0, 30, 30, 0])
    assert np.allclose(I, [1.7026057774431276,
                           0.7736072493369371,
                           0.7736072493369371,
                           1.7026057774431276])


def test_evaluation_double():
    """Test the map evaluation against some benchmarks [double]."""
    return run(multi=False)


def test_evaluation_multi():
    """Test the map evaluation against some benchmarks [multi]."""
    return run(multi=True)


if __name__ == "__main__":
    test_evaluation_double()
    test_evaluation_multi()
