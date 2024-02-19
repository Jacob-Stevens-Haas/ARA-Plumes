import numpy as np

from ara_plumes import models  # noqa: F401
from ara_plumes import utils  # noqa: F401


def test_circle_poly_intersection_real():
    x0 = np.sqrt(1 / 2 * (-1 + np.sqrt(5)))
    y0 = x0**2
    expected = np.array([[-x0, y0], [x0, y0]])
    result = models.PLUME.circle_poly_intersection(1, 0, 0, 1, 0, 0, True)
    np.testing.assert_array_almost_equal(result, expected)


def test_circle_poly_intersection_complex():
    x0 = np.sqrt(1 / 2 * (-1 + np.sqrt(5)))
    x1 = complex(0, np.sqrt(1 / 2 * (1 + np.sqrt(5))))
    y0 = x0**2
    y1 = x1**2
    expected = np.array([[-x0, y0], [-x1, y1], [x1, y1], [x0, y0]])
    result = models.PLUME.circle_poly_intersection(1, 0, 0, 1, 0, 0, False)
    np.testing.assert_array_almost_equal(result, expected)
