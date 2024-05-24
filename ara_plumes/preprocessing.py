import warnings
from typing import Union

import cv2
import IPython
import numpy as np
from tqdm import tqdm

from .typing import ColorVideo
from .typing import GrayImage
from .typing import GrayVideo


def convert_video_to_numpy_array(
    path_to_vid: str,
    start_frame: int = 0,
    end_frame: int = -1,
    gray: bool = True,
) -> Union[GrayVideo, ColorVideo]:
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
    gray : bool, optional (default True)
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
    if start_frame < 0:
        raise ValueError("start_frame must be int greater than or equal to 0.")

    # ensure end_frame is positive int less than tot_frame
    if end_frame == -1:
        end_frame = tot_frame
    elif end_frame > tot_frame:
        raise ValueError(
            f"end_frame must be int less than or equal to tot_frame count: {tot_frame}"
        )

    print(f"converting frames {start_frame} to {end_frame}")

    # convert frames to numpy arrays
    frames_as_arrays = []
    for k in tqdm(range(start_frame, end_frame)):

        # read frame
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame_k = vid_capture.read()

        # if frame cannot be read stop process and return array as is
        if not ret:
            warnings.warn(f"Frame {k} could not be read.\nHalting Process..")
            break

        # convert to gray
        if gray:
            frame_k = cv2.cvtColor(frame_k, cv2.COLOR_BGR2GRAY)

        frames_as_arrays.append(frame_k)

    return np.array(frames_as_arrays)


def clip_video(
    video_path: str,
    init_frame: int,
    fin_frame: int,
    extension: str = "mp4",
    save_path: str = "clipped_video",
    display_vid: bool = False,
):
    """
    Clip starting and ending portions of video and save as a new video.

    Parameters:
    ----------
    video_path: str
        path to video

    init_frame: int
        Starting frame of video, or number of frames from beginning
        to trim.

    fin_frame: int
        Ending frame, or last frame to keep in new video.
        All frames after are to be trimmed.

    extension: str (default 'mp4')
        video type to save new video as.

    save_path: str (default 'clipped_video')
        name/path to save new video in/as.

    display_vid: bool (default False)
        Display frames as trimming is being applied.

    """
    video = cv2.VideoCapture(video_path)
    print(type(video.read()))
    ret, frame = video.read()
    # return frame

    # grab video info for saving new file
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    frame_rate = int(video.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Possibly resave video
    clip_title = save_path + "." + extension
    out = cv2.VideoWriter(
        clip_title, fourcc, frame_rate, (frame_width, frame_height), 0
    )

    if display_vid:
        print("display_handle defined.")
        display_handle = IPython.display.display(None, display_id=True)

    # Loop results in the writing of B&W shortned video clip
    k = 0
    try:
        while ret:
            if k < fin_frame and k >= init_frame:
                # print("entered update")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out.write(frame)
                _, frame = cv2.imencode(".jpeg", frame)  # why is this jpeg?
                if display_vid:
                    # print("update display")
                    display_handle.update(IPython.display.Image(data=frame.tobytes()))
            print("k:", k)
            k += 1
            ret, frame = video.read()
    except KeyboardInterrupt:
        pass
    finally:
        print("finished")
        # video.release()
        out.release()
        if display_vid:
            display_handle.update(None)
    return


def _create_average_image_from_video(
    video_capture: cv2.VideoCapture, start_frame: int = 0, end_frame: int = -1
) -> GrayImage:
    """
    Creates average image from a specified window of frames from video.
    Parameters:
    ----------
    video_capture: cv2.VideoCapture
        Object after reading video file using opencv.

    start_frame: int (default 0)
        starting frame to create average image from.

    end_frame: int (default -1)
        ending frame to create average image from.

    Returns:
    -------
    background_img_np: (np.ndarray)
        Numpy array of average image (in grayscale).
    """
    tot_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0:
        raise ValueError("start_frame must be int greater than or equal to 0.")
    if end_frame == -1:
        end_frame = tot_frame
    elif end_frame > tot_frame:
        raise ValueError(
            f"end_frame must be int less than or equal to tot_frame count: {tot_frame}"
        )

    for k in range(start_frame, end_frame):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame_k = video_capture.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame: {k}")

        frame_k = cv2.cvtColor(frame_k, cv2.COLOR_BGR2GRAY)

        if k == start_frame:
            background_img_np = frame_k.astype(float)
        elif k > start_frame:
            background_img_np += frame_k.astype(float)

    img_count = end_frame - start_frame

    background_img_np = (background_img_np / img_count).astype(np.uint8)

    return background_img_np


def background_subtract(
    video_path: str,
    fixed_range: int,
    subtraction_method: str = "fixed",
    img_range=None,
    save_path: str = "subtraction",
    extension: str = "mp4",
    display_vid: bool = True,
):

    #####################
    # Fixed Subtraction #
    #####################

    video = cv2.VideoCapture(video_path)

    # Create background image
    if subtraction_method == "fixed" and (
        isinstance(fixed_range, int) or isinstance(fixed_range, list)
    ):
        print("Creating background image...")
        background_img_np = _create_average_image_from_video(
            video_capture=video,
            start_frame=fixed_range[0],
            end_frame=fixed_range[1],
        )
        # print("back shape:", background_img_np.shape)
        print("done.")
    else:
        raise TypeError(
            "fixed_range must be a positive int for fixed subtraction method."
        )
    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # grab video info for saving new file
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    frame_rate = int(video.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Possibly resave video
    if isinstance(save_path, str):
        clip_title = save_path + "." + extension
        color_true = 1
        out = cv2.VideoWriter(
            clip_title, fourcc, frame_rate, (frame_width, frame_height), color_true
        )

    if display_vid is True:
        display_handle = IPython.display.display(None, display_id=True)

    # Get image range to apply background subtraction
    if isinstance(img_range, list) and len(img_range) == 2:
        if img_range[-1] >= tot_frames:
            print("img_range exceeds total number of frames...")
            print(f"Using max frame count: {tot_frames}")
            init_frame = img_range[0]
            fin_frame = tot_frames - 1
        else:
            init_frame, fin_frame = img_range
    elif isinstance(img_range, int):
        init_frame = img_range
        fin_frame = tot_frames - 1
    elif img_range is None:
        if isinstance(fixed_range, int):
            init_frame = fixed_range
            fin_frame = tot_frames - 1
        elif isinstance(fixed_range, list) and len(fixed_range) == 2:
            init_frame = fixed_range[-1]
            fin_frame = tot_frames - 1
    else:
        raise TypeError(f"img_range does not support type {type(img_range)}.")

    # Frames to ignore
    for _ in range(init_frame):
        _ = video.read()

    # Frames to apply background subtraction too
    for k in tqdm(range(init_frame, fin_frame)):
        ret, frame = video.read()

        if not ret:
            print("break reached")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.subtract(frame, background_img_np)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if isinstance(save_path, str):
            out.write(frame)

        _, frame = cv2.imencode(".jpeg", frame)

        if display_vid is True:
            display_handle.update(IPython.display.Image(data=frame.tobytes()))

    if isinstance(save_path, str):
        video.release()
        out.release()
