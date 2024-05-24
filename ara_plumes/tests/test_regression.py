import numpy as np

from ..regressions import do_parametric_regression
from ..regressions import do_polynomial_regression
from ..regressions import do_sinusoid_regression


def test_do_polynomial_regression():
    expected = (1, 2, 3)
    poly_deg = 2
    a, b, c = expected

    def poly_func(x):
        return a * x**2 + b * x + c

    x = np.linspace(0, 1, 101)
    y = poly_func(x)

    step = 4
    result = do_polynomial_regression(x[::step], y[::step], poly_deg=poly_deg)
    np.testing.assert_almost_equal(expected, result)


def test_do_sinusoid_regression():
    expected = (1, 2, 3, 4)
    a, w, g, b = expected

    def sinusoid_func(t, r):
        return a * np.sin(w * r - g * t) + b * r

    axis = np.linspace(0, 1, 101)
    tt, rr = np.meshgrid(axis, axis)

    X = np.hstack((tt.reshape(-1, 1), rr.reshape(-1, 1)))
    Y = sinusoid_func(tt, rr).reshape(-1)

    step = 4
    result = do_sinusoid_regression(X[::step], Y[::step])

    np.testing.assert_array_almost_equal(expected, result)


def test_do_parametric_regression():
    expected1 = (1, 2, 3, 4)
    expected2 = (4, 3, 2, 1)

    a1, b1, c1, d1 = expected1
    a2, b2, c2, d2 = expected2

    def poly1(t):
        return a1 * t**3 + b1 * t**2 + c1 * t + d1

    def poly2(t):
        return a2 * t**3 + b2 * t**2 + c2 * t + d2

    X = np.linspace(0, 1, 101)
    Y = np.hstack((poly1(X).reshape(-1, 1), poly2(X).reshape(-1, 1)))

    result1, result2 = do_parametric_regression(X, Y, poly_deg=3)

    np.testing.assert_array_almost_equal(expected1, result1)

    np.testing.assert_array_almost_equal(expected2, result2)
