import starry
import theano
import theano.tensor as tt
import pytest
import numpy as np


@pytest.mark.xfail
def test_zero_lat_lon():
    """
    TODO: The gradient of the intensity is NaN when either the
    latitude or the longitude is zero. There's a div by zero in
    the Op that we need to work around.

    """
    map = starry.Map(ydeg=1)
    map[1, 0] = 0.5

    lat = tt.dscalar()
    lon = tt.dscalar()

    dIdlat = theano.function(
        [lat, lon], [tt.grad(map.intensity(lat=lat, lon=lon)[0], lat)]
    )
    dIdlon = theano.function(
        [lat, lon], [tt.grad(map.intensity(lat=lat, lon=lon)[0], lon)]
    )

    lat = 30
    lon = 30
    assert np.isfinite(dIdlat(lat, lon))
    assert np.isfinite(dIdlon(lat, lon))

    lat = 0
    lon = 30
    assert np.isfinite(dIdlat(lat, lon))
    assert np.isfinite(dIdlon(lat, lon))

    lat = 30
    lon = 0
    assert np.isfinite(dIdlat(lat, lon))
    assert np.isfinite(dIdlon(lat, lon))
