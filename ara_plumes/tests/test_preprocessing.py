import numpy as np

from .mock_video_utils import MockVideoCapture
from ara_plumes.preprocessing import _create_average_image_from_video


def test_create_average_image_from_video():
    video_capture_mock = MockVideoCapture()
    start_frame = 10
    end_frame = 20
    expected_avg_img = np.full((480, 640), 14, dtype=np.uint8)
    avg_img = _create_average_image_from_video(
        video_capture_mock, start_frame, end_frame
    )
    np.testing.assert_array_equal(avg_img, expected_avg_img)
