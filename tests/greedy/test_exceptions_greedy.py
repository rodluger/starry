# -*- coding: utf-8 -*-
"""
Test various exceptions throughout the code.

"""
import starry
import pytest


def test_negative_radius():
    map = starry.Map(ydeg=2)
    map[1, 0] = 1.0
    with pytest.raises(RuntimeError) as e:
        flux = map.flux(xo=0.99, yo=0, zo=1, ro=-1e-15)
    assert "Occultor radius is negative. Aborting." in str(e.value)


def test_bad_ylm_indices():
    map = starry.Map(ydeg=1, udeg=1)

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[2, 0]
    with pytest.raises(ValueError) as e:
        map[2, 0] = 1

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[-1, 0]
    with pytest.raises(ValueError) as e:
        map[-1, 0] = 1

    # Bad `m`
    x = map[1, 3]
    assert len(x) == 0
    # TODO: It would be nice if `map[1, 3] = ...` raised
    # an error, but currently it does nothing (silently).

    # Bad type
    with pytest.raises(ValueError) as e:
        x = map[2.3, 5.7]
    with pytest.raises(ValueError) as e:
        map[2.3, 5.7] = 1


def test_bad_wavelength_indices():
    map = starry.Map(ydeg=1, udeg=1, nw=2)

    # Bad index
    with pytest.raises(ValueError) as e:
        x = map[1, 0, -1]
    with pytest.raises(ValueError) as e:
        map[1, 0, -1] = 1

    # Bad type
    with pytest.raises(ValueError) as e:
        x = map[1, 0, "foo"]
    with pytest.raises(ValueError) as e:
        map[1, 0, "foo"] = 1


def test_bad_ul_indices():
    map = starry.Map(ydeg=1, udeg=1)

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[-1]
    with pytest.raises(ValueError) as e:
        map[-1] = 1

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[3]
    with pytest.raises(ValueError) as e:
        map[3] = 1

    # Bad type
    with pytest.raises(ValueError) as e:
        x = map[map]
    with pytest.raises(ValueError) as e:
        map[map] = 1


def test_bad_sys_settings():
    pri = starry.Primary(starry.Map())
    sec = starry.Secondary(starry.Map(), porb=1.0)

    # Bad exposure time
    with pytest.raises(AssertionError) as e:
        sys = starry.System(pri, sec, texp=-1.0)

    # Bad oversample factor
    with pytest.raises(AssertionError) as e:
        sys = starry.System(pri, sec, oversample=-1)

    # Bad integration order
    with pytest.raises(AssertionError) as e:
        sys = starry.System(pri, sec, order=99)

    # Bad primary
    with pytest.raises(AssertionError) as e:
        sys = starry.System(sec, sec)

    # Reflected light primary
    with pytest.raises(AssertionError) as e:
        sys = starry.System(starry.Primary(starry.Map(reflected=True)), sec)

    # Bad secondary
    with pytest.raises(AssertionError) as e:
        sys = starry.System(pri, pri)

    # No secondaries
    with pytest.raises(AssertionError) as e:
        sys = starry.System(pri)

    # Different number of wavelength bins
    with pytest.raises(AssertionError) as e:
        sys = starry.System(
            pri, starry.Secondary(starry.Map(ydeg=1, nw=2), porb=1.0)
        )

    # RV for secondary, but not primary
    with pytest.raises(AssertionError) as e:
        sys = starry.System(
            pri, starry.Secondary(starry.Map(rv=True), porb=1.0)
        )

    # Reflected light for first secondary, but not second
    with pytest.raises(ValueError) as e:
        sys = starry.System(
            pri, sec, starry.Secondary(starry.Map(reflected=True), porb=1.0)
        )


def test_bad_sys_data():
    pri = starry.Primary(starry.Map(ydeg=1))
    sec = starry.Secondary(starry.Map(ydeg=1), porb=1.0)
    sys = starry.System(pri, sec)

    # User didn't provide the covariance
    with pytest.raises(ValueError) as e:
        sys.set_data([0.0])

    # User didn't provide a dataset
    with pytest.raises(ValueError) as e:
        sys.solve(t=[0.0])

    # Provide a dummy dataset
    sys.set_data([0.0], C=1.0)

    # User didn't provide a prior for the primary
    with pytest.raises(ValueError) as e:
        sys.solve(t=[0.0])

    # Provide a prior for the primary
    pri.map.set_prior(L=1.0)

    # Provide a prior for the secondary
    sec.map.set_prior(L=1.0)

    # Now check that this works
    sys.solve(t=[0.0])


def test_bad_map_data():
    map = starry.Map(ydeg=1)

    # User didn't provide the covariance
    with pytest.raises(ValueError) as e:
        map.set_data([0.0])

    # User didn't provide a dataset
    with pytest.raises(ValueError) as e:
        map.solve()

    # Provide a dummy dataset
    map.set_data([0.0], C=1.0)

    # User didn't provide a prior
    with pytest.raises(ValueError) as e:
        map.solve()

    # Provide a prior
    map.set_prior(L=1.0)

    # Now check that this works
    map.solve()


def test_bad_map_types():
    # RV + Reflected
    with pytest.raises(NotImplementedError) as e:
        starry.Map(reflected=True, rv=True)

    # Limb-darkened + spectral
    with pytest.raises(NotImplementedError) as e:
        starry.Map(udeg=2, nw=10)
