"""
Test map rendering.

"""
import starry
import numpy as np
np.random.randn(41)


def test_reflected():
    lmax = 5
    map = starry.Map(lmax, reflected=True)
    map[1:, :] = np.random.randn((lmax + 1) ** 2 - 1)
    source = [-1, -0.5, 0.3]
    res = 50

    # Compute using `render`
    img1 = map.render(res=res, source=source)

    # Compute manually
    img2 = np.zeros((res, res))
    for i, x in enumerate(np.linspace(-1, 1, res)):
        for j, y in enumerate(np.linspace(-1, 1, res)):
            img2[j, i] = map(x=x, y=y, source=source)

    # Compute vectorized
    img3 = np.zeros((res, res))
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    x, y = np.meshgrid(x, y)
    img3 = map(x=x.flatten(), y=y.flatten(), source=source).reshape(res, res)

    assert np.allclose(img1, img2, equal_nan=True)
    assert np.allclose(img1, img3, equal_nan=True)


def test_emitted():
    lmax = 5
    map = starry.Map(lmax)
    map[1:, :] = np.random.randn((lmax + 1) ** 2 - 1)
    res = 50

    # Compute using `render`
    img1 = map.render(res=res)

    # Compute manually
    img2 = np.zeros((res, res))
    for i, x in enumerate(np.linspace(-1, 1, res)):
        for j, y in enumerate(np.linspace(-1, 1, res)):
            img2[j, i] = map(x=x, y=y)

    # Compute vectorized
    img3 = np.zeros((res, res))
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    x, y = np.meshgrid(x, y)
    img3 = map(x=x.flatten(), y=y.flatten()).reshape(res, res)

    assert np.allclose(img1, img2, equal_nan=True)
    assert np.allclose(img1, img3, equal_nan=True)