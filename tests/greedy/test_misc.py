# -*- coding: utf-8 -*-
"""
Miscellaneous tests.

"""
import starry
import io
import logging


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


def test_check_kwargs_body():
    starry.config.quiet = False
    with warnings.catch_warnings(record=True) as w:
        body = starry.Primary(starry.Map(), giraffe=10)
        assert len(w) == 1
        assert "Invalid keyword `giraffe`" in str(w[-1].message)
