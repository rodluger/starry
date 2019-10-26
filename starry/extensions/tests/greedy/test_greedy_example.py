import pytest
import starry

starry.config.lazy = False


def test_greedy_example():
    # Test greedy extensions here
    assert True


@pytest.mark.xfail
def test_greedy_example_failure():
    # Let's see what happens when an extension test fails
    assert False
