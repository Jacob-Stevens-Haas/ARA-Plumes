import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

from .mock_video_utils import MockVideoCapture
from ara_plumes import models
from ara_plumes import regressions
from ara_plumes import utils
from ara_plumes.models import create_average_image_from_numpy_array
from ara_plumes.models import create_average_image_from_video
from ara_plumes.preprocessing import convert_video_to_numpy_array


def test_create_background_img():
    # test numpy array
    plume = models.PLUME()
    plume.numpy_frames = np.array(
        [np.full((10, 10), i, dtype=np.uint8) for i in range(1, 11)]
    )

    expected_avg_img = np.full((10, 10), 5, dtype=np.uint8)
    avg_img = plume.create_background_img(img_range=10)

    np.testing.assert_array_equal(avg_img, expected_avg_img)

    # test video
    plume = models.PLUME()
    plume.video_capture = MockVideoCapture()

    expected_avg_img = np.full((480, 640), 14, dtype=np.uint8)
    avg_img = plume.create_background_img(img_range=[10, 20])

    np.testing.assert_array_equal(avg_img, expected_avg_img)

    # test Attribute Error
    plume = models.PLUME()
    with pytest.raises(AttributeError):
        plume.create_background_img(img_range=[10, 20])


def test_create_average_image_from_video():
    video_capture_mock = MockVideoCapture()
    start_frame = 10
    end_frame = 20
    expected_avg_img = np.full((480, 640), 14, dtype=np.uint8)
    avg_img = create_average_image_from_video(
        video_capture_mock, start_frame, end_frame
    )
    np.testing.assert_array_equal(avg_img, expected_avg_img)


def test_create_average_image_from_numpy_array():
    expected_avg_img = np.full((10, 10), 5, dtype=np.uint8)
    arr_of_frames = np.array(
        [np.full((10, 10), i, dtype=np.uint8) for i in range(1, 11)]
    )
    avg_img = create_average_image_from_numpy_array(arr_of_frames)
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
