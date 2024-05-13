from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


def convert_video_to_numpy_array(
    path_to_vid: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    gray: bool = True,
) -> Union[
    np.ndarray[tuple[Frame, Width, Height], np.dtype[np.uint8]],
    np.ndarray[tuple[Frame, Width, Height, Channel], np.dtype[np.uint8]]
]:
    """
    Convert a video file to a NumPy array of frames.

    Parameters:
    ----------
    path_to_vid : str
        Path to the video file.
    start_frame : int, optional (default 0)
        Starting frame index. If not provided, defaults to the first frame (0).
    end_frame : int, optional (default total frame)
        Ending frame index. If not provided, defaults to the last frame of the
        video.
    gray : bool, optional
        Flag to convert frames to grayscale. If True, frames will be converted
        to grayscale. Default is True.

    Returns:
    -------
    np.ndarray
        NumPy array containing the frames of the video.
        Each frame is represented as a NumPy array.
    """

    # get video capture object
    vid_capture = cv2.VideoCapture(path_to_vid)

    # get total number of frames
    tot_frame = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # ensure start_frames is positive int
    if not start_frame:
        start_frame = 0
    elif not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame must be int greater than or equal to 0.")

    # ensure end_frame is positive int less than tot_frame
    if not end_frame:
        end_frame = tot_frame
    elif not isinstance(end_frame, int) or end_frame > tot_frame:
        raise ValueError(
            f"end_frame must be int less than or equal to tot_frame count: {tot_frame}"
        )

    print(f"converting frames {start_frame} to {end_frame-1}")

    # convert frames to numpy arrays
    frames_as_arrays = []
    for k in tqdm(range(start_frame, end_frame)):

        # read frame
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame_k = vid_capture.read()

        # if frame cannot be read stop process and return array as is
        if not ret:
            print(f"Frame {k} could not be read.\nHaulting Process..")
            break

        # convert to gray
        if gray:
            frame_k = cv2.cvtColor(frame_k, cv2.COLOR_BGR2GRAY)

        frames_as_arrays.append(frame_k)

    return np.array(frames_as_arrays)
