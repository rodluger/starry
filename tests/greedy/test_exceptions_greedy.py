# -*- coding: utf-8 -*-
"""
Test various exceptions throughout the code.

"""
import starry
import pytest


def test_lazy_change():
    map = starry.Map()
    with pytest.raises(Exception) as e:
        starry.config.lazy = True
    assert "Cannot change the `starry` config at this time." in str(e.value)


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

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[-1, 0]

    # Bad `m`
    x = map[1, 3]
    assert len(x) == 0

    # Bad type
    with pytest.raises(ValueError) as e:
        x = map[2.3, 5.7]


def test_bad_wavelength_indices():
    map = starry.Map(ydeg=1, udeg=1, nw=2)

    # Bad index
    with pytest.raises(ValueError) as e:
        x = map[1, 0, -1]

    # Bad type
    with pytest.raises(ValueError) as e:
        x = map[1, 0, "foo"]


def test_bad_ul_indices():
    map = starry.Map(ydeg=1, udeg=1)

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[-1]

    # Bad `l`
    with pytest.raises(ValueError) as e:
        x = map[3]

    # Bad type
    with pytest.raises(ValueError) as e:
        x = map[map]


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
