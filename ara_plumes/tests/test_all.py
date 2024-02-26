import numpy as np

from ara_plumes import models  # noqa: F401
from ara_plumes import regressions
from ara_plumes import utils


def test_circle_poly_intersection_real():
    x0 = np.sqrt(1 / 2 * (-1 + np.sqrt(5)))
    y0 = x0**2
    expected = np.array([[-x0, y0], [x0, y0]])
    result = utils.circle_poly_intersection(1, 0, 0, 1, 0, 0, True)
    np.testing.assert_array_almost_equal(result, expected)


def test_circle_poly_intersection_complex():
    x0 = np.sqrt(1 / 2 * (-1 + np.sqrt(5)))
    x1 = complex(0, np.sqrt(1 / 2 * (1 + np.sqrt(5))))
    y0 = x0**2
    y1 = x1**2
    expected = np.array([[-x0, y0], [-x1, y1], [x1, y1], [x0, y0]])
    result = utils.circle_poly_intersection(1, 0, 0, 1, 0, 0, False)
    np.testing.assert_array_almost_equal(result, expected)


def test_regression_sinusoid():
    regression_func = regressions.regression

    # Generate data
    A, w, g, B = (-0.1, 10, 4.14159265, 1)

    def sinusoid_func(t, x):
        return A * np.sin(w * x - g * t) + B * x

    x = np.linspace(0, 1, 101)
    t = np.array([1 for _ in range(len(x))])
    Y = sinusoid_func(t, x)
    X = np.hstack((t.reshape(-1, 1), x.reshape(-1, 1)))

    expected = (A, w, g, B)
    result = regression_func(X, Y, regression_method="sinusoid")

    np.testing.assert_almost_equal(expected, result)


def test_regression_poly():
    regression_func = regressions.regression

    def poly_func(x):
        return x**2

    x = np.linspace(0, 1, 101)
    y = poly_func(x)

    expected = (1, 0, 0)
    result = regression_func(X=x, Y=y, regression_method="poly")
    np.testing.assert_almost_equal(expected, result)


def test_circle_intersection():
    x0 = 0
    y0 = 0
    r0 = np.sqrt(2)
    x1 = 1
    y1 = 0
    r1 = 1
    expected = np.array([[1, 1], [1, -1]])
    result = utils.circle_intersection(x0=x0, y0=y0, r0=r0, x1=x1, y1=y1, r1=r1)
    np.testing.assert_almost_equal(expected, result)
