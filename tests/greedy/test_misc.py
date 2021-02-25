# -*- coding: utf-8 -*-
"""
Miscellaneous tests.

"""
import starry
import io
import logging
import warnings


def test_quiet():
    """Test quiet setting."""
    # Get root logger
    logger = logging.getLogger()

    # Assert we get a message
    starry.config.quiet = False
    with io.StringIO() as capture:
        ch = logging.StreamHandler(capture)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        map = starry.Map(ydeg=1)
        assert "Pre-computing some matrices" in capture.getvalue()
        logger.removeHandler(ch)

    # Assert we don't
    starry.config.quiet = True
    with io.StringIO() as capture:
        ch = logging.StreamHandler(capture)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        map = starry.Map(ydeg=1)
        assert len(capture.getvalue()) == 0
        logger.removeHandler(ch)


def test_check_kwargs_map(caplog):
    """Test that we capture bad keyword arguments."""
    starry.config.quiet = False
    caplog.clear()
    map = starry.Map(giraffe=10)
    assert len(caplog.records) >= 1
    assert any(
        [
            "Invalid keyword `giraffe`" in str(rec.message)
            for rec in caplog.records
        ]
    )


def test_check_kwargs_map_flux(caplog):
    """Test that we capture bad keyword arguments."""
    starry.config.quiet = False
    caplog.clear()
    flux = starry.Map().flux(giraffe=10)
    assert len(caplog.records) >= 1
    assert any(
        [
            "Invalid keyword `giraffe`" in str(rec.message)
            for rec in caplog.records
        ]
    )


def test_check_kwargs_body(caplog):
    """Test that we capture bad keyword arguments."""
    starry.config.quiet = False
    caplog.clear()
    body = starry.Primary(starry.Map(), giraffe=10)
    assert len(caplog.records) >= 1
    assert any(
        [
            "Invalid keyword `giraffe`" in str(rec.message)
            for rec in caplog.records
        ]
    )


def test_coefficient_numbers():
    map = starry.Map(ydeg=2, udeg=3)
    assert map.ydeg == 2
    assert map.udeg == 3
    assert map.fdeg == 0
    assert map.deg == 5
    assert map.Ny == 9
    assert map.Nu == 4
    assert map.Nf == 1
    assert map.N == 36
