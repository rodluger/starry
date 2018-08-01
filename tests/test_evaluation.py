"""Test the map evaluation."""
from starry2 import Map
import numpy as np


def test_evaluation():
    """Test the map evaluation with gradients."""
    # Instantiate
    map = Map()
    map[:] = 1

    # No arguments
    I = map.evaluate()
    assert np.allclose(I, 1.4014804341818383)

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1)
    assert np.allclose(I, 1.7026057774431276)

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1, theta=30)
    assert np.allclose(I, 0.7736072493369371)

    # Vector evaluation
    I = map.evaluate(x=[0.1, 0.2, 0.3], y=[0.1, 0.2, 0.3], theta=30)
    assert np.allclose(I, [0.7736072493369371,
                           1.0432785526935853,
                           1.318434613210305])

    # Rotation caching
    I = map.evaluate(x=0.1, y=0.1, theta=[0, 30, 30, 0])
    assert np.allclose(I, [1.7026057774431276,
                           0.7736072493369371,
                           0.7736072493369371,
                           1.7026057774431276])


if __name__ == "__main__":
    test_evaluation()
