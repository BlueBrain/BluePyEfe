from bluepyefe.ecode import tools
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal


def test_scipy_signal2d():
    np.random.seed(42)
    data = np.random.uniform(-1, 1, 10)
    res = tools.scipy_signal2d(data, 85)
    assert_array_almost_equal(
        res,
        [
            0.2022300234864176,
            0.1973169683940732,
            0.1973169683940732,
            0.1973169683940732,
            0.1973169683940732,
            0.1973169683940732,
            0.1973169683940732,
            0.2022300234864176,
            0.2022300234864176,
            0.2022300234864176,
        ],
    )


def test_base_current():
    np.random.seed(42)
    data = np.random.uniform(-1, 1, 10)
    base = tools.base_current(data)
    assert_almost_equal(base, 0.1973169683940732)
