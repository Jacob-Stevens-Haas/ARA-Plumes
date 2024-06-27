import logging
import warnings
from typing import Literal
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian

from . import regressions
from . import utils
from .concentric_circle import concentric_circle
from .regressions import regress_frame_mean
from .typing import AX_FRAME
from .typing import Bool1D
from .typing import ColorImage
from .typing import Contour_List
from .typing import Float1D
from .typing import Float2D
from .typing import Frame
from .typing import GrayImage
from .typing import GrayVideo
from .typing import List
from .typing import NpFlt
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
    def clean_video(
        arr: GrayVideo,
        fixed_range: tuple[int, int] = (0, -1),
        gauss_space_blur: bool = True,
        gauss_time_blur: bool = True,
        gauss_space_kws: Optional[dict] = None,
        gauss_time_kws: Optional[dict] = None,
    ) -> GrayVideo:
        """
        Apply fixed background subtraction and gaussian bluring
        to GrayVideo.

        Parameters:
        ----------
        arr:
            Array of frames to apply background subtraction and blurring
            to.

        fixed_range:
            Range of images to use as background image for subtraction
            method.

        gauss_space_blur: bool (default True)
            Apply GaussianBlur in space.

        gauss_time_blur: bool (default True)
            Apply Gaussian blur across time series

        gauss_space_kws:
            dictionary for keyword arguments for apply_gauss_space_blur function.

        gauss_time_kws:
            dictionary for keyword arguments for apply_gauss_time_blur function.

        Returns:
        -------
        clean_vid:
            Frames with background subtraction and blurring applied.
        """
        if len(arr.shape) == 4:
            raise TypeError("arr must be be in gray.")

        print("applying background subtract:")
        clean_vid = background_subtract(frames=arr, fixed_range=fixed_range)

        if gauss_space_blur:
            print("applying space blur:")
            if gauss_space_kws is None:
                gauss_space_kws = {}
            clean_vid = apply_gauss_space_blur(arr=clean_vid, **gauss_space_kws)

        if gauss_time_blur:
            print("applying time blur:")
            if gauss_time_kws is None:
                gauss_time_kws = {}
            clean_vid = apply_gauss_time_blur(arr=clean_vid, **gauss_time_kws)

        return clean_vid

    @staticmethod
    def video_to_ROM(
        arr: GrayVideo,
        orig_center: tuple[int, int],
        img_range: tuple[int, int] = (0, -1),
        concentric_circle_kws: Optional[dict] = None,
        get_contour_kws: Optional[dict] = None,
    ) -> tuple[
        List[tuple[Frame, PlumePoints]],
        List[tuple[Frame, PlumePoints]],
        List[tuple[Frame, PlumePoints]],
    ]:
        """
        Apply connetric circles to frames range.

        Parameters:
        -----------
        arr:
            Array of frames to apply method to.

        orig_center:
            tuple declaring plume leak source location.

        img_range: list (default None)
            Range of images to apply background subtraction method and
            concentric circles too. Default uses all frames from arr.

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
        mean_points = []
        var1_points = []
        var2_points = []

        start_frame, end_frame = img_range
        if end_frame == -1:
            end_frame = len(arr)

        print("applying concentric circles:")
        for k, frame_k in enumerate(arr[start_frame:end_frame]):
            k += start_frame
            selected_contours = get_contour(frame_k, **get_contour_kws)

            mean_k, var1_k, var2_k = concentric_circle(
                frame_k,
                contours=selected_contours,
                orig_center=orig_center,
                **concentric_circle_kws,
            )

            mean_points.append((k, mean_k))
            var1_points.append((k, var1_k))
            var2_points.append((k, var2_k))

        return mean_points, var1_points, var2_points

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

        for i, (_, frame_points) in enumerate(mean_points):

            if decenter:
                frame_points[:, 1:] -= decenter
            try:
                coef_time_series[i] = regress_frame_mean(
                    frame_points, regression_method, poly_deg
                )
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"Insufficient training points in frame {i}", stacklevel=2
                )
                coef_time_series[i] = [np.nan for _ in range(n_coef)]

        return coef_time_series

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

    selected_contours = [c.reshape(-1, 2) for c in selected_contours]

    return selected_contours


def apply_gauss_space_blur(
    arr: GrayVideo,
    kernel_size: tuple[int, int] = (81, 81),
    sigma_x: float = 15,
    sigma_y: float = 15,
    iterative: bool = True,
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

    if iterative:
        blurred_arr = np.empty_like(arr)

        for i, frame in enumerate(arr):
            blurred_arr[i] = cv2.GaussianBlur(
                frame, ksize=kernel_size, sigmaX=sigma_x, sigmaY=sigma_y
            )

    else:
        blurred_arr = np.array(
            [
                cv2.GaussianBlur(
                    frame_i, ksize=kernel_size, sigmaX=sigma_x, sigmaY=sigma_y
                )
                for frame_i in arr
            ],
            dtype=np.uint8,
        )

    return blurred_arr


def apply_gauss_time_blur(
    arr: GrayVideo, kernel_size: int = 5, sigma: float = 1, iterative: bool = True
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
        raise ValueError(
            "Kernel size for Gaussian blur must be a positive, odd integer."
        )
    gw = gaussian(kernel_size, sigma)
    gw = gw / np.sum(gw)
    if iterative:
        blurred_arr = np.zeros_like(arr, dtype=np.uint8)

        for i in range(arr.shape[AX_FRAME]):

            start_idx = max(i - kernel_size // 2, 0)
            end_idx = min(i + kernel_size // 2 + 1, arr.shape[AX_FRAME])

            slice_arr = arr[start_idx:end_idx]
            convolved_slice = (
                convolve1d(slice_arr, gw, axis=AX_FRAME, mode="constant", cval=0.0)
            ).astype(np.uint8)

            slice_index = min(kernel_size // 2, len(convolved_slice))
            if i < kernel_size:
                slice_index = -1 * (slice_index + 1)

            blurred_arr[i] = convolved_slice[slice_index]

    else:
        blurred_arr = (
            convolve1d(arr, gw, axis=AX_FRAME, mode="constant", cval=0.0)
        ).astype(np.uint8)

    return blurred_arr


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


def _create_background_img(frames: GrayVideo, img_range: tuple[int, int]) -> Float2D:
    """
    Create background image for fixed subtraction method.
    Args:
        frames: Video to create background image
        img_range: Range of frames to create average image (list).
        Or number of initial frames to create average image to subtract from
        frames (int).
    Returns:
        np.ndarray: Numpy array of average image (in
        grayscale).
    """

    if isinstance(img_range, int):
        start_frame = 0
        end_frame = img_range
    else:
        start_frame, end_frame = img_range

    if end_frame == -1:
        end_frame = len(frames)

    return np.mean(frames[start_frame:end_frame], axis=AX_FRAME)


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


def get_contour_list(
    clean_vid: GrayVideo,
    threshold_method: str = "OTSU",
    num_of_contours: int = 1,
    contour_smoothing: bool = False,
    contour_smoothing_eps: int = 50,
    find_contour_method: int = cv2.CHAIN_APPROX_NONE,
    decenter: Optional[tuple[int,int]]=None
) -> List[Contour_List]:
    """
    Return contours learned from frames of clean video.

    Parameters:
    ----------
    clean_vid:
        frames of gray video.

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
    
    decenter:
        To decenter founds contours so plume leak source is at origin `(0,0)`.

    Returns:
    --------
    cont_list:
        Returns list of selected contours from each frame of clean_vid.

    """
    def _decenter_selected_contours(selected_contours,decenter):
        selected_contours_origin = []
        for cont in selected_contours:
            selected_contours_origin.append(cont-decenter)
        return selected_contours_origin

    cont_list = []
    for frame in clean_vid:
        selected_contours = get_contour(
            frame,
            threshold_method=threshold_method,
            num_of_contours=num_of_contours,
            contour_smoothing=contour_smoothing,
            contour_smoothing_eps=contour_smoothing_eps,
            find_contour_method=find_contour_method,
        )

        if decenter is not None:
            selected_contours = _decenter_selected_contours(
                selected_contours, decenter
            )
            
        cont_list.append(selected_contours)

    return cont_list


def flatten_var_points(
    coef_timeseries: Float2D,
    vari_points: List[PlumePoints],
    selected_contours: Contour_List,
    regression_method: str,
) -> Float2D:
    """
    Convert edge coordinates, `(r,x,y)`, of plume learned at different frames
    into `L2` distances from mean regression path at time t, `(t,r,d)`.

    Parameters:
    ----------
    coef_timeseries:
        Array of learend mean path polynomial coefficients, in descedning order of
        degree, for each frame.

    vari_points:
        timestamp t, and (r,x,y) coordinate for edge of plume at time t.

    selected_contours:
        list of plume contours identified for each frame.

    regression_method:
        Method used to create mean path regression.

    Returns:
    -------
    np.ndarray:
        Array of coordinates, (t,r,d), of L2 distance, `d`, of mean regression path to
        edge coordinate along concentric circle with radii `r` at time frame `t`.


    Pseudo Example:
    -------
    >>> coef_timeseries = np.array([
                [a0,b0,c0],
                [a1,b1,c1]
        ])
    >>> vari_points = [
            (t0, np.array([[r0,x0,y0],[r1,x1,y0]]))
            (t1, np.array([[R0,X0,Y0],[R1,X1,Y1]]))
        ]
    >>> selected_contours = <list of contours>
    >>> trd_arr = flatten_var_points(
                     coef_timeseries,
                     vari_points,
                     'poly',
                     selected_contours
                  )
    >>> trd_arr
    >>> np.array([
            [t0,r0,d0]
            [t0,r1,d1],
            [t1,R0,D0],
            [t1,R1,D1]
        ])

    """
    arr = None
    for coef, vari, cont in zip(coef_timeseries, vari_points, selected_contours):
        trd_arr = _convert_rxy_to_trd(coef, vari, cont, regression_method)
        if arr is None:
            arr = trd_arr
        else:
            arr = np.vstack((arr, trd_arr))

    return arr


def _convert_rxy_to_trd(
    coef: Float1D,
    vari_points: tuple[int, PlumePoints],
    selected_contours: Contour_List,
    regression_method: str,
) -> np.ndarray[tuple[int, Literal[3]], NpFlt]:
    """
    Converts coordinates taken of edge model at time t to flatten p_mean space.
    Takes elements in array vari_points, [r_0(t), x_0(t), y_0(t)], and converts
    to flattened mean regression space of [t, r_0(t), d_0(t)] where d(t) is
    L2 distance from the mean regression function along the concentric circle
    with radii r(t).

    Parameters:
    ----------
    coef: Array of polynomial coefficients in descending order of degree
          for mean path.


    vari_points: Timestamp t and (r,x,y) coordinates of edge model at timestamp t.
        >>> vari_points = (t, np.array([
                                [r_0, x_0, y_0],
                                [r_1, x_1, y_1],
                                 ...
                                [r_l, x_l, y_l]
                              ]))


    selected_contours: List of contours.

    regression_method: method used to create mean path polynomial. 'linear', 'poly',
        'poly_inv', and 'poly_para'.

    Returns:
    -------
    np.ndarray: Array of points [t,r_i(t), d_i(t)].
            >>> np.array([
                    [t, r_0, d_0],
                    [t, r_1, d_1],
                    ...
                    [t, r_l, d_l]
            ])

    """
    if regression_method == "poly" or regression_method == "linear":
        return _poly_rxy_to_trd(coef, vari_points, selected_contours)

    if regression_method == "poly_inv":
        return _poly_rxy_to_trd(coef, vari_points, selected_contours, inv=True)

    if regression_method == "poly_para":
        return _poly_para_rxy_to_trd(coef, vari_points)


def _poly_para_rxy_to_trd(coef, vari_points):
    """
    convert vari_points to flattened p_mean points where coef is from parametric
    polynomial.
    """
    mid_index = len(coef) // 2
    f1 = np.polynomial.Polynomial(coef[:mid_index][::-1])
    f2 = np.polynomial.Polynomial(coef[mid_index:][::-1])

    t, points = vari_points
    trd_arr = []
    for r, x, y in points:
        d = np.linalg.norm(np.array((f1(r), f2(r))) - (x, y))
        trd_arr.append((t, r, d))

    return np.array(trd_arr)


def _poly_rxy_to_trd(coef, vari_points, selected_contours, inv=False):
    """
    convert vari_points to flattened p_mean points where coef is from explicit
    polynomial.
    """
    t, points = vari_points
    x0, y0 = points[0][1:]

    trd_arr = [(t, 0, 0)]
    for r, x, y in points[1:, :]:
        sols = utils.circle_poly_intersection(r=r, x0=x0, y0=y0, poly_coef=coef[::-1])
        if inv:
            if len(sols.shape) == 2:
                sols = sols[:, ::-1]
            else:
                sols = sols[::-1]
        sols = sols[_sol_in_contour(sols, selected_contours)]
        d = np.linalg.norm(sols - (x, y))
        trd_arr.append((t, r, d))

    return np.array(trd_arr)


def _sol_in_contour(
    sols: List[tuple[int, int]], selected_contours: Contour_List
) -> Bool1D:
    """
    Checks if points lie within any set of contours.

    Parameters:
    -----------
    sols: List of (x,y) coordinates to check
    selected_contours: List of contours

    Returns:
    -------
    Bool1D: 1d array of bool vals specifying which points in sol lie within selected
            contours.
    """
    mask = []
    for sol in sols:
        in_arr = False
        for contour in selected_contours:
            if cv2.pointPolygonTest(contour, sol, False) == 1:
                in_arr = True
        mask.append(in_arr)

    return np.array(mask)
