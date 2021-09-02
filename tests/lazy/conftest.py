# -*- coding: utf-8 -*-
import pytest
import starry
import matplotlib


@pytest.fixture(scope="module", autouse=True)
def setup():
    matplotlib.use("Agg")
    starry.config.lazy = True
    yield
