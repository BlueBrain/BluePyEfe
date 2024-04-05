"""conftest.py contains fixtures that are automatically imported by pytest."""
import pytest
import matplotlib


@pytest.fixture(autouse=True, scope='session')
def set_matplotlib_backend():
    matplotlib.use('Agg')  # to avoid opening windows during testing
