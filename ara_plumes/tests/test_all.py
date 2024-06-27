import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from ..concentric_circle import concentric_circle
from ..models import _poly_para_rxy_to_trd
from ..models import _poly_rxy_to_trd
from ..models import _sol_in_contour
from ..models import flatten_var_points
from ..models import get_contour_list
from ..utils import _square_poly_coef
from ..utils import circle_intersection
from ..utils import circle_poly_intersection
from ara_plumes import models
from ara_plumes import regressions
from ara_plumes.models import apply_gauss_space_blur
from ara_plumes.models import apply_gauss_time_blur
from ara_plumes.models import get_contour
from ara_plumes.preprocessing import convert_video_to_numpy_array


test_numpy_frames = np.array(
    [np.full((10, 10), i, dtype=np.uint8) for i in range(1, 11)]
)


def test_concentric_circle():
    # create data
    scale = 4
    orig_center = (50, 200)
    height, width = 400, 400
    bw_img = np.zeros((height, width), dtype=np.uint8)

    bw_img[25:376, 50] = 255 // scale
    bw_img[25, 50:251] = 255 / scale
    bw_img[375, 50:251] = 255 // scale
    bw_img[25:376, 250] = 255 // scale

    num_of_circs = 3
    r = 50
    for i in range(num_of_circs):
        i += 2
        bw_img[200, r * i] = 255

    rectangle_plume = bw_img.copy()

    # get result
    selected_contours = get_contour(rectangle_plume)
    result_mean, result_var1, result_var2 = concentric_circle(
        rectangle_plume,
        selected_contours=selected_contours,
        orig_center=orig_center,
        radii=r,
        num_of_circs=num_of_circs,
    )

    # expected data
    expected_mean = np.array(
        [[0, 50, 200], [50, 100, 200], [100, 150, 200], [150, 200, 200]]
    )
    expected_var1 = np.array(
        [[0, 50, 200], [50, 50, 250], [100, 50, 300], [150, 50, 350]]
    )
    expected_var2 = np.array(
        [[0, 50, 200], [50, 50, 150], [100, 50, 100], [150, 50, 50]]
    )

    np.testing.assert_array_equal(expected_mean, result_mean)
    np.testing.assert_array_equal(expected_var1, result_var1)
    np.testing.assert_array_equal(expected_var2, result_var2)


def test_get_contour_list():
    width, height = 400, 400
    square_img = np.zeros((width, height), dtype=np.uint8)
    square_img[100:301, 100:301] = 255
    clean_vid = np.array([square_img, square_img])
    expected = np.array([[100, 100], [100, 300], [300, 300], [300, 100]])
    expected = [expected, expected]
    result = get_contour_list(clean_vid, find_contour_method=2)
    for e, r in zip(expected, result):
        np.testing.assert_array_almost_equal(e, r[0])
    
def test_get_contour_list_decenter():
    width, height = 400, 400
    square_img = np.zeros((width, height), dtype=np.uint8)
    square_img[100:301, 100:301] = 255
    clean_vid = np.array([square_img, square_img])
    expected = np.array([[0, 0], [0, 200], [200, 200], [200, 0]])
    expected = [expected, expected]
    result = get_contour_list(clean_vid, find_contour_method=2,decenter=(100,100))
    for e, r in zip(expected, result):
        np.testing.assert_array_almost_equal(e, r[0])

def test_get_contour():
    width, height = 400, 400
    square_img = np.zeros((width, height), dtype=np.uint8)
    square_img[100:301, 100:301] = 255
    square_img_color = cv2.cvtColor(square_img, cv2.COLOR_GRAY2BGR)
    expected_big = np.array(
        [[100, 100], [100, 300], [300, 300], [300, 100]], dtype=np.int32
    )
    expected_small = np.array([[25, 25], [25, 50], [50, 50], [50, 25]], dtype=np.int32)

    # gray test
    selected_contour = get_contour(square_img, find_contour_method=2)
    result_big = selected_contour[0]
    np.testing.assert_array_equal(expected_big, result_big)

    # color test
    selected_contour = get_contour(square_img_color, find_contour_method=2)
    result_big = selected_contour[0]
    np.testing.assert_array_equal(expected_big, result_big)

    # two contours
    two_square_img = square_img.copy()
    two_square_img[25:51, 25:51] = 255
    selected_contour = get_contour(
        two_square_img, num_of_contours=2, find_contour_method=2
    )
    result_big, result_small = selected_contour
    np.testing.assert_array_almost_equal(expected_big, result_big)
    np.testing.assert_array_almost_equal(expected_small, result_small)


def test_gauss_space_blur():
    expected = test_numpy_frames
    result_iter = apply_gauss_space_blur(
        test_numpy_frames, kernel_size=(3, 3), sigma_x=1, sigma_y=1, iterative=True
    )
    result = apply_gauss_space_blur(
        test_numpy_frames.astype(float),
        kernel_size=(3, 3),
        sigma_x=1,
        sigma_y=1,
        iterative=False,
    )

    np.testing.assert_array_equal(expected, result_iter)
    np.testing.assert_array_equal(expected, result)


def test_gauss_time_blur():
    ksize = 3
    n_frames = 10
    # test_numpy_frames = test_numpy_frames.astype(float)
    result_iter = apply_gauss_time_blur(
        test_numpy_frames, kernel_size=ksize, sigma=np.inf, iterative=True
    )
    result = apply_gauss_time_blur(
        test_numpy_frames, kernel_size=ksize, sigma=np.inf, iterative=False
    )

    def expected_seq(k, ksize=ksize, n_frames=n_frames):
        def sum_int_i_to_j_and_divide_ksize(i, j):
            return np.sum([np.round(k / ksize) for k in range(i, j + 1)])

        r = int(ksize / 2)
        i = k - r
        j = k + r

        if j > n_frames:
            j = n_frames
        if i > n_frames:
            i = n_frames + 1
        if j <= 0:
            j = 1
        if i <= 0:
            i = 1

        return sum_int_i_to_j_and_divide_ksize(i, j)

    expected = np.array(
        [np.full((10, 10), expected_seq(i), dtype=np.uint8) for i in range(1, 11)]
    )
    expected[6, :, :] = 6
    np.testing.assert_array_equal(expected, result)
    np.testing.assert_array_equal(expected, result_iter)


def test_create_background_img():
    # test numpy array
    expected_avg_img = np.full((10, 10), 5.5)
    avg_img = models._create_background_img(test_numpy_frames, img_range=(0, 10))

    np.testing.assert_array_equal(avg_img, expected_avg_img)


def test_background_subtract():
    expected = np.concatenate(
        (
            np.zeros((5, 10, 10), dtype=np.uint8),
            np.array([np.full((10, 10), i, dtype=np.uint8) for i in range(1, 6)]),
        ),
        axis=0,
    )
    result = models.background_subtract(test_numpy_frames, fixed_range=(0, 10))

    np.testing.assert_array_equal(result, expected)


def test_convert_video_to_numpy_array():
    # Mock cv2.VideoCapture and its methods
    mock_capture = MagicMock()
    mock_capture.get.return_value = 100  # Mock total frame count
    mock_capture.read.side_effect = [
        (True, np.ones((100, 100, 3), dtype=np.uint8)) for _ in range(100)
    ]  # Mock reading frames

    # Mock cv2.VideoCapture constructor
    with unittest.mock.patch("cv2.VideoCapture", return_value=mock_capture):
        # Call the function with mocked arguments
        result = convert_video_to_numpy_array(
            "dummy_video.mp4", start_frame=0, end_frame=10, gray=True
        )

    # Assertions
    # Check number of frames
    np.testing.assert_equal(len(result), 10)
    # Check if frames are numpy arrays
    assert all(isinstance(frame, np.ndarray) for frame in result)

    # Check frame shape (grayscale)
    np.testing.assert_equal(result[0].shape, (100, 100))


def test_circle_poly_intersection():
    # constant
    coef = [0]
    r = 1
    x0 = 0
    y0 = 0

    expected = np.array([[-1, 0], [1, 0]])

    result = circle_poly_intersection(r, x0, y0, coef)

    np.testing.assert_array_almost_equal(expected, result)

    # linear
    coef = (1, 0)
    r = 1
    x0 = 0
    y0 = 0
    expected = np.array(
        [[-1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]]
    )
    result = circle_poly_intersection(r, x0, y0, coef[::-1])
    np.testing.assert_array_almost_equal(expected, result)

    # quadratic
    coef = (1, 0, 0)
    y = (-1 + np.sqrt(5)) / 2

    expected = np.array([[-np.sqrt(y), y], [np.sqrt(y), y]])
    result = circle_poly_intersection(r, x0, y0, coef[::-1])
    np.testing.assert_array_almost_equal(expected, result)

    # cubic
    coef = (1, 0, -1, 0)
    expected = np.array([[-1, 0], [1, 0]])
    result = circle_poly_intersection(r, x0, y0, coef[::-1])
    np.testing.assert_array_almost_equal(expected, result)


def test_square_poly_coef():
    coef = (1, 0, 1)
    expected = np.array([1, 0, 2, 0, 1])
    result = _square_poly_coef(coef)
    np.testing.assert_array_equal(expected, result)

    coef = (1, 0, 2)
    expected = np.array([1, 0, 4, 0, 4])
    result = _square_poly_coef(coef)
    np.testing.assert_array_equal(expected, result)


def test_edge_regression_sinusoid():
    regression_func = regressions.edge_regression

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


def test_circle_intersection():
    x0 = 0
    y0 = 0
    r0 = np.sqrt(2)
    x1 = 1
    y1 = 0
    r1 = 1
    expected = np.array([[1, 1], [1, -1]])
    result = circle_intersection(x0=x0, y0=y0, r0=r0, x1=x1, y1=y1, r1=r1)
    np.testing.assert_almost_equal(expected, result)


def test_flatten_var_points():
    t = 0
    points = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]])
    vari = (t, points)
    vari_points = [vari, vari]
    coef_timeseries = np.array([[1, 0], [1, 0]])

    def f(x, r):
        return np.sqrt((x - r / np.sqrt(2)) ** 2 + r**2 / 2)

    expected = np.array([[0, 0, 0], [0, 1, f(1, 1)], [0, 2, f(2, 2)], [0, 3, f(3, 3)]])
    expected = np.vstack((expected, expected))
    length = 5
    selected_contours = [
        np.array(
            [[[0, length]], [[0, 0]], [[length, 0]], [[length, length]]], dtype=np.int32
        )
    ]
    selected_contours = [selected_contours, selected_contours]

    result = flatten_var_points(
        coef_timeseries,
        vari_points,
        regression_method="poly",
        selected_contours=selected_contours,
    )
    np.testing.assert_array_almost_equal(expected, result)


def test_poly_para_rxy_to_trd():
    t = 0
    points = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]])
    vari = (t, points)
    coef = np.array([1, 0, 1, 0]) / np.sqrt(2)
    result = _poly_para_rxy_to_trd(coef, vari)

    def f(x, r):
        return np.sqrt((x - r / np.sqrt(2)) ** 2 + r**2 / 2)

    expected = np.array([[0, 0, 0], [0, 1, f(1, 1)], [0, 2, f(2, 2)], [0, 3, f(3, 3)]])
    np.testing.assert_array_almost_equal(expected, result)


def test_poly_rxy_to_trd():
    t = 0
    points = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]])
    vari = (t, points)
    coef = np.array([1, 0])
    length = 5
    selected_contours = [
        np.array(
            [[[0, length]], [[0, 0]], [[length, 0]], [[length, length]]], dtype=np.int32
        )
    ]

    result = _poly_rxy_to_trd(
        coef=coef, vari_points=vari, selected_contours=selected_contours
    )

    def f(x, r):
        return np.sqrt((x - r / np.sqrt(2)) ** 2 + r**2 / 2)

    expected = np.array([[0, 0, 0], [0, 1, f(1, 1)], [0, 2, f(2, 2)], [0, 3, f(3, 3)]])

    np.testing.assert_array_almost_equal(expected, result)


def test_sol_in_contour():
    contour = [np.array([[[0, 2]], [[0, 0]], [[2, 0]], [[2, 2]]], dtype=np.int32)]
    sols = [[1, 1], [3, 3]]
    result = _sol_in_contour(sols=sols, selected_contours=contour)
    expected = np.array([True, False])
    np.testing.assert_array_equal(expected, result)
