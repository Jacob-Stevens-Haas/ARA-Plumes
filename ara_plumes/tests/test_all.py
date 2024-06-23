import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from ..concentric_circle import concentric_circle
from ara_plumes import models
from ara_plumes import regressions
from ara_plumes import utils
from ara_plumes.models import _create_average_image_from_numpy_array
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


def test_get_contour():
    width, height = 400, 400
    square_img = np.zeros((width, height), dtype=np.uint8)
    square_img[100:301, 100:301] = 255
    square_img_color = cv2.cvtColor(square_img, cv2.COLOR_GRAY2BGR)
    expected = np.array([[[100, 100]], [[100, 300]], [[300, 300]], [[300, 100]]])

    # gray test
    selected_contour = get_contour(square_img, find_contour_method=2)
    result = selected_contour[0]
    np.testing.assert_array_equal(expected, result)

    # color test
    selected_contour = get_contour(square_img_color, find_contour_method=2)
    result = selected_contour[0]
    np.testing.assert_array_equal(expected, result)


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
        1.0 * test_numpy_frames, kernel_size=ksize, sigma=np.inf, iterative=False
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


def test_create_average_image_from_numpy_array():
    expected_avg_img = np.full((10, 10), 5.5)
    # arr_of_frames = np.array(
    #     [np.full((10, 10), i, dtype=np.uint8) for i in range(1, 11)]
    # )
    avg_img = _create_average_image_from_numpy_array(test_numpy_frames)
    np.testing.assert_array_equal(avg_img, expected_avg_img)


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


def test_edge_regression_poly():
    regression_func = regressions.edge_regression

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
