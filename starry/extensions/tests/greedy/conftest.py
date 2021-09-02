# -*- coding: utf-8 -*-
import pytest
import starry


@pytest.fixture(scope="module", autouse=True)
def setup():
    starry.config.lazy = False
    yield
