import pytest


@pytest.fixture
def something():
    """
    Functions with @pytest.fixture run before each test and can be added as arguments to other tests
    """
    return 2


def test_example(something):
    """
    All tests should start with `test_`
    Test files should also start with `test_`
    """
    assert 1 + 1 == something
