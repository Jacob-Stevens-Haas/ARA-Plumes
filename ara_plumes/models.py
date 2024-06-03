import logging
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian
from tqdm import tqdm

from . import regressions
from . import utils
from .concentric_circle import concentric_circle
from .regressions import regress_frame_mean
from .typing import AX_FRAME
from .typing import ColorImage
from .typing import Contour_List
from .typing import Float2D
from .typing import FloatImage
from .typing import Frame
from .typing import GrayImage
from .typing import GrayVideo
from .typing import List
from .typing import PlumePoints
from .typing import X_pos
from .typing import Y_pos

logger = logging.getLogger(__name__)


class PLUME:
    def __init__(self):
        self.mean_poly = None
        self.var1_poly = None
        self.var2_poly = None
        self.orig_center = None
        self.var1_dist = []
        self.var2_dist = []
        self.var1_params_opt = None
        self.var2_params_opt = None
        self.var1_func = None
        self.var2_func = None

    def read_video(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_width = int(self.video_capture.get(3))
        self.frame_height = int(self.video_capture.get(4))
        self.fps = int(self.video_capture.get(5))
        self.tot_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_numpy_arr(self, numpy_frames):
        self.numpy_frames = numpy_frames

    def display_frame(self, frame: int):
        cap = self.video_capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()

        if ret:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

    def set_center(self, frame: Optional[int] = None) -> None:
        """Set the plume source in the video"""
        self.orig_center = click_coordinates(self.video_capture, frame)

    @staticmethod
    def regress_multiframe_mean(
        mean_points: List[tuple[Frame, PlumePoints]],
        regression_method: str,
        poly_deg: int = 2,
        decenter: Optional[tuple[X_pos, Y_pos]] = None,
    ) -> Float2D:
        """
        Converts plumepoints into timeseries of regressed coefficients.

        Parameters:
        ----------
        mean_points:
            List of tuples returned from PLUME.train()

        regression_method:
            Regression methods to apply to arr.
            'linear':    Applies explicit linear regression to (x,y)
            'poly':      Applies explicit polynomial regression to (x,y) with degree
                        up to poly_deg
            'poly_inv':  Applies explcity polynomial regression to (y,x) with degree
                        up to poly_deg
            'poly_para': Applies parametric poly regression to (r,x) and (r,y) with
                        degree up to poly_deg
        poly_deg:
            degree of regression for all poly methods. Note 'linear' ignores this
            argument.

        decenter:
            Tuple to optionally subtract from from points prior to regression

        Returns:
        -------
        coef_time_series:
            Returns np.ndarray of regressed coefficients.
        """
        n_frames = len(mean_points)

        if regression_method == "poly_para":
            n_coef = 2 * (poly_deg + 1)
        elif regression_method == "linear":
            n_coef = 2
        else:
            n_coef = poly_deg + 1

        coef_time_series = np.zeros((n_frames, n_coef))

        for i, (_, frame_points) in tqdm(enumerate(mean_points)):

            if decenter:
                frame_points[:, 1:] -= decenter

            coef_time_series[i] = regress_frame_mean(
                frame_points, regression_method, poly_deg
            )

        return coef_time_series

    def train(
        self,
        img_range: tuple[int, int] = (0, -1),
        fixed_range: tuple[int, int] = (0, -1),
        gauss_space_blur: bool = True,
        gauss_kernel_size: int = 81,
        gauss_space_sigma: float = 15,
        gauss_time_blur: bool = True,
        gauss_time_window: bool = 5,
        gauss_time_sigma: float = 1,
        concentric_circle_kws={},
        get_contour_kws={},
    ) -> tuple[
        List[tuple[Frame, PlumePoints]],
        List[tuple[Frame, PlumePoints]],
        List[tuple[Frame, PlumePoints]],
    ]:
        """
        Apply connetric circles to frames range.

        Parameters:
        -----------
        img_range: list (default None)
            Range of images to apply background subtraction method and
            concentric circles too. Default None uses remaining frames
            after declared fixed_range used to generated background
            image.

        fixed_range:
            Range of images to use as background image for subtraction
            method.

        gauss_space_blur: bool (default True)
            Apply GaussianBlur in space.

        gauss_kernel_size: odd int (default 81)
            Size of kernel for GaussianBlur. Must be odd int.

        gauss_space_sigma: int (default 15)

        gauss_time_blur: bool (default True)

        gauss_time_window: int (default 5)

        gauss_time_sigma: int (default 1)

        concentric_circle_kws:
            dictionary containing arguments for concentric_circle function.

        get_contour_kws:
            dictionary contain arguments for get_contour function

        Returns:
        --------
        mean_points:
            List of tuples containing frame count, k, and mean points, (r(k),x(k),y(k)),
            attained from frame k.

        var1_points:
            List of tuples containing frame count, k, and var1 points, (r(k),x(k),y(k)),
            attained from frame k.

        var2_points:
            List of tuples containing frame count, k, and var2 points, (r(k),x(k),y(k)),
            attained from frame k.

        """
        if hasattr(self, "numpy_frames"):
            if len(self.numpy_frames.shape) == 4:
                raise TypeError("numpy_frames must be be in gray.")
            print("applying background subtract")
            clean_vid = background_subtract(
                frames=self.numpy_frames, fixed_range=fixed_range, img_range=img_range
            )
            print("subtraction done.\n")
        else:
            raise AttributeError("PLUME object must read in a numpy array of frames.")

        if gauss_time_blur:
            print("applying time blur")
            clean_vid = apply_gauss_time_blur(
                arr=clean_vid, kernel_size=gauss_time_window, sigma=gauss_time_sigma
            )
            print("time blur done.\n")

        if gauss_space_blur:
            print("applying space blur")
            clean_vid = apply_gauss_space_blur(
                arr=clean_vid,
                kernel_size=(gauss_kernel_size, gauss_kernel_size),
                sigma_x=gauss_space_sigma,
                sigma_y=gauss_space_sigma,
            )
            print("space blur done.\n")

        mean_points = []
        var1_points = []
        var2_points = []

        start_frame, end_frame = img_range
        if end_frame == -1:
            end_frame = len(clean_vid)

        for k, frame_k in enumerate(tqdm(clean_vid)):
            k += start_frame
            selected_contours = get_contour(frame_k, **get_contour_kws)

            mean_k, var1_k, var2_k = concentric_circle(
                frame_k,
                selected_contours=selected_contours,
                orig_center=self.orig_center,
                **concentric_circle_kws,
            )

            mean_points.append((k, mean_k))
            var1_points.append((k, var1_k))
            var2_points.append((k, var2_k))

        return mean_points, var1_points, var2_points

    def train_variance(self, kernel_fit=False):
        """
        Learned sinusoid coefficients (A_opt, w_opt, g_opt, B_opt) for variance data
        on flattened p_mean, vari_dist, attained from PLUME.train().
        """

        #############
        # Var1_dist #
        #############

        # preprocess data
        var1_dist = self.var1_dist

        # flatten var1_dist
        var1_txy = regressions.flatten_vari_dist(var1_dist)

        # split into training and test data (no normalization)
        train_index = int(len(var1_dist) * 0.8)

        X = var1_txy[:, :2]
        Y = var1_txy[:, 2]

        X_train = X[:train_index]
        X_test = X[train_index:]
        Y_train = Y[:train_index]
        Y_test = Y[train_index:]

        # apply ensemble learning
        var1_param_opt, var1_param_hist = regressions.var_ensemble_learn(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            n_samples=int(len(X_train) * 0.8),
            trials=2000,
            kernel_fit=kernel_fit,
        )
        print("var1 opt params:", var1_param_opt)
        self.var1_params_opt = var1_param_opt

        # Create var1_func
        var1_func = var_func_on_poly()
        var1_func.var_on_flattened_poly_params = var1_param_opt
        var1_func.orig_center = self.orig_center
        var1_func.poly_func_params = self.mean_poly

        self.var1_func = var1_func

        #############
        # Var2_dist #
        #############

        # preprocess data
        var2_dist = self.var2_dist

        # flatten var1_dist
        var2_txy = regressions.flatten_vari_dist(var2_dist)

        # split into training and test data (no normalization)
        train_index = int(len(var2_dist) * 0.8)

        X = var2_txy[:, :2]
        Y = var2_txy[:, 2]

        X_train = X[:train_index]
        X_test = X[train_index:]
        Y_train = Y[:train_index]
        Y_test = Y[train_index:]

        # apply ensemble learning
        var2_param_opt, var2_param_hist = regressions.var_ensemble_learn(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            n_samples=int(len(X_train) * 0.8),
            trials=2000,
            kernel_fit=kernel_fit,
        )

        self.var2_params_opt = var2_param_opt
        print("var2 opt params:", var2_param_opt)

        var2_func = var_func_on_poly()
        var2_func.var_on_flattened_poly_params = var2_param_opt
        var2_func.orig_center = self.orig_center
        var2_func.poly_func_params = self.mean_poly
        var2_func.upper_lower_envelope = "lower"

        self.var2_func = var2_func

        return var1_param_opt, var2_param_opt

    def plot_ROM_plume(self, t, show_plot=True):
        """
        Plot a single frame, at time t, of the ROM plume
        """
        var1_func = self.var1_func
        var2_func = self.var2_func
        a, b, c = self.mean_poly[t]

        def poly_func(x):
            return a * x**2 + b * x + c

        x_range = self.frame_width
        y_range = self.frame_height

        # Polynomial
        # TO DO: shift x0? - DONE
        x = np.linspace(0, self.orig_center[0], 200) - self.orig_center[0]

        # TO DO: add y0 back in - DONE
        y_poly = poly_func(x) + self.orig_center[1]

        # var
        var1_to_plot = [var1_func.eval_x(t, i) for i in x]
        var2_to_plot = [var2_func.eval_x(t, i) for i in x]

        # remove none type for plotting
        # TO DO: rescale back on  - DONE
        var1_to_plot = (
            np.array([x for x in var1_to_plot if x is not None]) + self.orig_center
        )
        var2_to_plot = (
            np.array([x for x in var2_to_plot if x is not None]) + self.orig_center
        )

        # generate plot
        # TO DO: shift by original center i.e., add + x0 - DONE
        plt.clf()
        plt.scatter(self.orig_center[0], self.orig_center[1], c="red")
        plt.plot(x + self.orig_center[0], y_poly, label="mean poly", c="red")
        plt.plot(var1_to_plot[:, 0], var1_to_plot[:, 1], label="var1", c="blue")
        plt.plot(var2_to_plot[:, 0], var2_to_plot[:, 1], label="var2", c="blue")
        plt.title(f"ROM Plume, t={t}")
        plt.legend(loc="upper right")

        # fix frame
        plt.xlim(0, x_range)
        plt.ylim(y_range, 0)
        if show_plot is True:
            plt.show()


class var_func_on_poly:
    """
    Function class to translated learned variances on flattened p_mean to
    an unflattened p_mean. i.e., the regular cartesian grid.

    Parameters:
    -----------
    var_on_flattened_poly_params: tuple
        Learned parameters for training varaince on flattened p_mean.
        (A_opt, w_opt, g_opt, B_opt).

    poly_func_params: np.ndarray
        Numpy array contained the learned coefficients of mean polynomial for
        each time frame

    orig_center: tuple
        The original plume leak source (x,y)

    upper_lower_envelope: str (default "upper")
        declares whether to plot unflattened varaince above p_mean ("upper")
        or below p_mean ("lower")
    """

    def __init__(self) -> None:
        self.var_on_flattened_poly_params = None
        self.poly_func_params = None
        self.orig_center = None
        self.upper_lower_envelope = "upper"

    def poly_func(self, t, x):
        a, b, c = self.poly_func_params[t]
        y = a * x**2 + b * x + c
        return y

    def sinusoid_func(self, t, r):
        A_opt, w_opt, g_opt, B_opt = self.var_on_flattened_poly_params
        d = A_opt * np.sin(w_opt * r - g_opt * t) + B_opt * r
        return d

    def eval_x(self, t, x):
        # get r on flattened p_mean plane
        x1 = x
        x0, y0 = (0, 0)  # TO DO: make this x0 = y0 = 0 - DONE
        y1 = self.poly_func(t, x1)

        r = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        # get d on flattened p_mean plane
        d = self.sinusoid_func(t, r)

        # get y on regular cartesian plane
        sols = utils.circle_intersection(x0=x0, y0=y0, r0=r, x1=x1, y1=y1, r1=d)

        if self.upper_lower_envelope == "upper":
            for sol in sols:
                if sol[1] >= self.poly_func(t, sol[0]):
                    return sol
        elif self.upper_lower_envelope == "lower":
            for sol in sols:
                if sol[1] <= self.poly_func(t, sol[0]):
                    return sol


def get_contour(
    img: GrayImage | ColorImage,
    threshold_method: str = "OTSU",
    num_of_contours: int = 1,
    contour_smoothing: bool = False,
    contour_smoothing_eps: int = 50,
    find_contour_method: int = cv2.CHAIN_APPROX_NONE,
) -> Contour_List:
    """
    Contour detection applied to single frame of background subtracted
    video.

    Parameters:
    -----------
    img: np.ndarray (gray or BGR)
        image to apply contour detection too.

    threshold_method: str (default "OTSU")
        Opencv method used for applying thresholding.

    num_of_contours: int (default 1)
        Number of contours selected to be used (largest to smallest).

    contour_smoothing: bool (default False)
        Used cv2.approxPolyDP to apply additional smoothing to contours
        selected contours.

    contour_smoothing_eps: positive int (default 50)
        hyperparater for tuning cv2.approxPolyDP in contour_smoothing.
        Level of smoothing to be applied to plume detection contours.
        Only used when contour_smoothing = True.

    find_contour_method:
        Method used by opencv to find contours. 1 is cv2.CHAIN_APPROX_NONE.
        2 is cv2.CHAIN_APPROX_SIMPLE.

    Returns:
    --------
    selected_contours:
        Returns list of num_of_contours largest contours detected in image.
    """

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img.copy()
    else:
        raise TypeError("img must be either Gray or Color")

    # Apply thresholding
    if threshold_method == "OTSU":
        _, threshold = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Find contours
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, method=find_contour_method
    )

    # Select n largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    selected_contours = contours[:num_of_contours]

    # Apply contour smoothing
    if contour_smoothing:
        smoothed_contours = []

        for contour in selected_contours:
            smoothed_contours.append(
                cv2.approxPolyDP(contour, contour_smoothing_eps, True)
            )

        selected_contours = smoothed_contours

    return selected_contours


def _create_average_image_from_numpy_array(arr: GrayVideo) -> FloatImage:
    """
    Creates average frame from numpy array ofr frames.
    Parameters:
    ----------
    arr: np.ndarray
        numpy array that contains all frames to average

    Returns:
    -------
    np.ndarray:
        average image created.
    """
    return np.mean(arr, axis=AX_FRAME)


def apply_gauss_space_blur(
    arr: GrayVideo,
    kernel_size: tuple[int, int],
    sigma_x: float,
    sigma_y: float,
) -> GrayVideo:
    """
    Apply openCV gaussianblur to series of frames

    Parameters:
    ----------
    arr: np.ndarray
    kernel_size:
        Gaussian convolution width and height. int(s) must be positive and odd and
        may differ.
    sigma_x:
    sigma_y:
    """

    blur_video = np.empty_like(arr)

    for i, frame in enumerate(arr):
        blur_video[i] = cv2.GaussianBlur(frame, kernel_size, sigma_x, sigma_y)

    return blur_video


def apply_gauss_time_blur(
    arr: GrayVideo,
    kernel_size: int,
    sigma: float,
) -> GrayVideo:
    """
    Applying Gaussian blur across timeseries of frames.

    Parameters:
    ----------
    arr:
        np.ndarray containing black and white frames.
    kernel_size:
        Positive, odd, int specifying the size of kernel to convolve with arr.
    sigma:
        standard deviation for Gaussian weighting.

    """
    if kernel_size % 2 == 0:
        raise TypeError("window for gaussian blur must be positive, odd int.")
    gw = gaussian(kernel_size, sigma)

    return (
        convolve1d(arr, gw, axis=AX_FRAME, mode="constant", cval=0.0) / np.sum(gw)
    ).astype(np.uint8)


def click_coordinates(
    vid: cv2.VideoCapture, frame: Optional[int] = None
) -> tuple[float, float]:
    """
    Scan to a frame and get coordinates of user click.

    Can be used to identify the plume source in video coordinates.

    Args:
        frame: Which frame to select from. Default is None, which
            gives middle frame of video.

    Returns:
        User click in video_coordinates

    Raises:
        ValueError: If frame has no data or cannot reach frame
        RuntimeError: If user input is closed externally
    """

    if frame is None:
        tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = tot_frames // 2

    # get frame image
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(frame):
        vid.grab()
    found_frame, image = vid.retrieve()
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not found_frame:
        raise ValueError(f"Cannot get image from frame {frame}")
    logger.debug("made it to pre selected points")

    # allow users to select point
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    fig.show()
    try:
        selected_point = plt.ginput(n=1)[0]
    except IndexError:
        raise RuntimeError("Point not selected")
    plt.close(fig)
    logger.debug("Point-selection input closed")
    if not selected_point:
        raise RuntimeError("Point not selected")
    return selected_point


def _create_background_img(frames: GrayVideo, img_range: tuple[int, int]):
    """
    Create background image for fixed subtraction method.
    Args:
        frames: Video to create background image
        img_range: Range of frames to create average image (list).
        Or number of initial frames to create average image to subtract from
        frames (int).
    Returns:
        background_img_np (np.ndarray): Numpy array of average image (in
        grayscale).
    """

    if isinstance(img_range, int):
        start_frame = 0
        end_frame = img_range
    else:
        start_frame, end_frame = img_range

    if end_frame == -1:
        end_frame = len(frames)

    background_img_np = _create_average_image_from_numpy_array(
        arr=frames[start_frame:end_frame]
    )

    return background_img_np


def background_subtract(
    frames: GrayVideo,
    fixed_range: tuple[int, int],
    img_range: tuple[int, int] = (0, -1),
) -> GrayVideo:
    """
    Applies fixed background subtraction to series of frames.

    Parameters:
    ----------
    frames:
        frames to use for subtraction method and generating background img.

    fixed_range:
        index slice to use on frames arg to create background img used for
        subtraction.

    img_range:
        index slice to use on frames arg to apply subtraction to.

    Returns:
    -------
    clean_vid:
        Series of frames with background subtraction applied.
    """
    background_img_np = _create_background_img(frames, img_range=fixed_range).astype(
        np.uint8
    )

    start_frame, end_frame = img_range
    if end_frame == -1:
        end_frame = len(frames)

    clean_vid = np.empty(
        shape=(end_frame - start_frame, frames[0].shape[0], frames[0].shape[1]),
        dtype=frames[0].dtype,
    )

    for i, frame in enumerate(frames[start_frame:end_frame]):
        clean_vid[i] = cv2.subtract(frame, background_img_np)

    return clean_vid
