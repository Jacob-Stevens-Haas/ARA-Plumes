import numpy as np

from ..models import PLUME
from ..regressions import do_parametric_regression
from ..regressions import do_polynomial_regression
from ..regressions import do_sinusoid_regression
from ..regressions import regress_frame_mean


def test_regress_multiframe_mean():
    regress_multiframe_mean = PLUME.regress_multiframe_mean
    expected = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    a, b, c, d = expected[0]
    e, f, g, h = expected[1]

    def poly1(t):
        return a * t**3 + b * t**2 + c * t + d

    def poly2(t):
        return e * t**3 + f * t**2 + g * t + h

    slice = 4
    R = np.linspace(0, 1, 101)

    mean_points_1 = np.vstack((R, R, poly1(R))).T
    mean_points_2 = np.vstack((R, R, poly2(R))).T

    mean_points = [(1, mean_points_1[::slice, :]), (2, mean_points_2[::slice, :])]

    result = regress_multiframe_mean(mean_points, "poly", 3)

    np.testing.assert_array_almost_equal(expected, result)

    # test decenter
    origin = (10, 10)
    mean_points_1[:, 1:] += origin
    mean_points_2[:, 1:] += origin
    mean_points = [(1, mean_points_1[::slice, :]), (2, mean_points_2[::slice, :])]

    result = regress_multiframe_mean(mean_points, "poly", 3, decenter=origin)
    np.testing.assert_array_almost_equal(expected, result)

    # test poly_para
    mean_points_1 = np.vstack((R, poly1(R), poly2(R))).T
    mean_points_2 = np.vstack((R, poly2(R), poly1(R))).T

    mean_points = [(1, mean_points_1[::slice, :]), (2, mean_points_2[::slice, :])]
    result = regress_multiframe_mean(mean_points, "poly_para", 3)

    np.testing.assert_array_almost_equal(
        np.hstack((expected[0], expected[1])), result[0]
    )
    np.testing.assert_array_almost_equal(
        np.hstack((expected[1], expected[0])), result[1]
    )

    # test poly_para nan
    mean_points = [(1, mean_points_1[::slice, :]), (2, mean_points_2[0, :])]
    result = regress_multiframe_mean(mean_points, "poly_para", 3)

    np.testing.assert_array_almost_equal(
        np.hstack((expected[0], expected[1])), result[0]
    )
    np.testing.assert_array_almost_equal(
        np.array([np.nan for _ in range(8)]), result[1]
    )


def test_regress_mean_points_k():
    # linear
    slice = 4
    R = np.linspace(0, 1, 101)
    mean_points = np.vstack((R, R, R)).T

    expected = (1, 0)
    result = regress_frame_mean(mean_points[::slice, :], method="linear")
    np.testing.assert_array_almost_equal(expected, result)

    # poly
    expected = (1, 2, 3)
    a, b, c = expected

    def poly_func(x):
        return a * x**2 + b * x + c

    R = np.linspace(0, 1, 101)
    mean_points = np.vstack((R, R, poly_func(R))).T

    result = regress_frame_mean(mean_points[::slice, :], method="poly")

    np.testing.assert_array_almost_equal(expected, result)

    # poly_inv
    mean_points = np.vstack((R, poly_func(R), R)).T

    result = regress_frame_mean(mean_points[::slice, :], method="poly_inv")

    np.testing.assert_array_almost_equal(expected, result)

    # poly_para
    expected = (1, 2, 3, 4, 4, 3, 2, 1)

    a, b, c, d, e, f, g, h = expected

    def poly1(t):
        return a * t**3 + b * t**2 + c * t + d

    def poly2(t):
        return e * t**3 + f * t**2 + g * t + h

    R = np.linspace(0, 1, 101)

    mean_points = np.vstack((R, poly1(R), poly2(R))).T

    result = regress_frame_mean(mean_points[::slice, :], method="poly_para", poly_deg=3)

    np.testing.assert_array_almost_equal(expected, result)


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
    expected = (1, 2, 3, 4, 4, 3, 2, 1)

    a, b, c, d, e, f, g, h = expected

    def poly1(t):
        return a * t**3 + b * t**2 + c * t + d

    def poly2(t):
        return e * t**3 + f * t**2 + g * t + h

    X = np.linspace(0, 1, 101)
    Y = np.hstack((poly1(X).reshape(-1, 1), poly2(X).reshape(-1, 1)))

    result = do_parametric_regression(X, Y, poly_deg=3)

    np.testing.assert_array_almost_equal(expected, result)
