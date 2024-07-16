from typing import cast

import cv2
import numpy as np

from .typing import Bool1D
from .typing import ColorImage
from .typing import Contour_List
from .typing import Float2D
from .typing import GrayImage
from .typing import PlumePoints
from .typing import X_pos
from .typing import Y_pos


def concentric_circle(
    img: GrayImage | ColorImage,
    contours: Contour_List,
    orig_center: tuple[X_pos, Y_pos],
    radii: int = 50,
    num_of_circs: int = 30,
    interior_scale: float = 3 / 5,
    rtol: float = 1e-2,
    atol: float = 1e-6,
    quiet: bool = True,
) -> tuple[PlumePoints, PlumePoints, PlumePoints]:
    """
    Applies concentric cirlces to a single frame (gray or BGR) from video

    Creates new image and learned poly coef for mean line and variance line.

    Parameters:
    -----------
    img:
        image to apply concentric circle too.

    selected_contours:
        List of contours learned from get_contour

    orig_center:
        (x,y) coordinates of plume leak source.

    radii:
        The radii used to step out in concentric circles method.

    num_of_circles:
        number of circles and radii steps to take in concentric circles method.

    interior_scale:
        Used to scale down the radii used on the focusing rings. Called in
        find_next_center

    rtol, atol:
        Relative and absolute tolerances. Used in np.isclose function in
        find_max_on_boundary and find_next_center functions.
        Checks if points are close to selected radii.

    quiet:
        suppresses error output


    Returns:
    --------
    points_mean:
        Returns nx3 array containing observed points along mean path.
        Where the kth entry is of the form [r(k), x(k), y(k)], i.e the
        coordinate (x,y) of the highest value point along the concetric circle
        with radii r(k).

    points_var1:
        Returns nx3 array containing observed points along upper envolope path,
        i.e., above the mean path. The kth entry is of the form [r(k), x(k), y(k)],
        i.e the coordinate (x,y) of the intersection with the plume contour along
        the concentric circle with raddi r(k).

    points_var2:
        Returns nx3 array containing observed points along lower envolope path,
        i.e., below the mean path. The kth entry is of the form [r(k), x(k), y(k)],
        i.e the coordinate (x,y) of the intersection with the plume contour along
        the concentric circle with raddi r(k).
    """

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    img_gray = cast(GrayImage, img_gray)

    points_mean = _apply_concentric_search(
        img_gray, num_of_circs, orig_center, radii, interior_scale, rtol, atol, quiet
    )

    # Limit points to those within contours, plus origin #
    xy_points_no_origin = cast(Float2D, points_mean[1:, 1:])
    mask = _sol_in_contours(xy_points_no_origin, contours)
    points_in_cont = points_mean[1:][mask]

    if len(points_in_cont) == 0:
        points_mean = cast(PlumePoints, points_mean[:1])
    else:
        points_mean = cast(
            PlumePoints, np.vstack((points_mean[:1], points_mean[1:][mask]))
        )

    contour_dists = _contour_distances(contours, orig_center)

    points_var1, points_var2 = _get_edge_points(
        num_of_circs, radii, contour_dists, orig_center
    )

    if len(points_var1) == 0:
        points_var1 = cast(PlumePoints, points_mean[:1])
    else:
        points_var1 = cast(PlumePoints, np.vstack((points_mean[:1], points_var1)))

    if len(points_var2) == 0:
        points_var2 = cast(PlumePoints, points_mean[:1])
    else:
        points_var2 = cast(PlumePoints, np.vstack((points_mean[:1], points_var2)))

    return points_mean, points_var1, points_var2


def _get_edge_points(
    num_of_circs: int,
    radii: float,
    contours: list[PlumePoints],
    orig_center: tuple[float, float],
) -> tuple[PlumePoints, PlumePoints]:
    """
    Find edge points based on largest and smallest polar angle on a given radii.
    """
    cw_edge_points = []
    ccw_edge_points = []

    for step in range(1, num_of_circs + 1):
        radius = step * radii
        contour_crosses = _find_intersections(contours, radius)
        if len(contour_crosses) <= 1:
            continue

        edge_candidates = _interpolate_intersections(
            contour_crosses, radius, orig_center
        )
        polar_candidates = _append_polar_angle(edge_candidates, orig_center)

        max_idx = np.argmax(polar_candidates[:, -1])
        min_idx = np.argmin(polar_candidates[:, -1])

        cw_edge_points.append(polar_candidates[min_idx][:-1])
        ccw_edge_points.append(polar_candidates[max_idx][:-1])

    return np.array(ccw_edge_points), np.array(cw_edge_points)


def _find_intersections(contours: list[PlumePoints], radius: float) -> set[PlumePoints]:
    """
    Find pairs of points where they cross over certain radii value.

    NOTE: In edge case where a point is exactly equal to radius, it is treated as
          greater than radius.
    """
    pairs_of_points = []
    for contour in contours:
        sign = contour[-1, 0] < radius
        for point_idx, (cur_radius, *_) in enumerate(contour):
            if sign ^ (cur_radius < radius):
                pairs_of_points.append(
                    np.vstack((contour[point_idx - 1], contour[point_idx]))
                )
                sign = not sign

    return pairs_of_points


def _interpolate_intersections(
    contour_crosses: set[PlumePoints], radius: float, orig_center: tuple[float, float]
) -> PlumePoints:
    """
    Apply linear interpolation to set of points where radius is
    independent variable. Interpolation done via barycentric coordinates.

    For sets of points `(x0,y0)` and `(x1,y1)` and original center `(cx,cy)`,
    we find `a` such that

     `||f(a) - (cx,cy)||^2_2` = radius

     where,
     `f(a):=(x,y)= a*(x0,y0) + (1-a)*(x1,y1)`

     Parameters:
     ----------
     contours_crosses:
        pairs of all points (r,x,y) that cross over radius.

    radius:
        radius of circle from original center

    orig_center:
        coorindate of center of circle.

    Returns:
    -------
    PlumePoints
    """

    def bary_interpolation(x0, y0, x1, y1):
        def bary_func(alpha):
            return alpha * np.array([x0, y0]) + (1 - alpha) * np.array([x1, y1])

        return bary_func

    def opt_alpha(x0, y0, x1, y1, cx, cy, r):
        """return a that solves `||f(a) - (cx,cy)||**2 = radius`"""
        denominator = (x0 - x1) ** 2 + (y0 - y1) ** 2
        term1 = (cx - x1) * (x0 - x1)
        term2 = (cy - y1) * (y0 - y1)

        inner_term1 = -(cy**2) * (x0 - x1) ** 2
        inner_term2 = r**2 * ((x0 - x1) ** 2 + (y0 - y1) ** 2)
        inner_term3 = 2 * cy * (x0 - x1) * (-x1 * y0 + cx * (y0 - y1) + x0 * y1)
        inner_term4 = (cx * y0 - x1 * y0 - cx * y1 + x0 * y1) ** 2

        sqrt_term = np.sqrt(inner_term1 + inner_term2 + inner_term3 - inner_term4)

        numerator = term1 + term2 - sqrt_term

        alpha = numerator / denominator

        return alpha

    inter_points = []
    cx, cy = orig_center
    for pairs in contour_crosses:
        r0, x0, y0 = pairs[0]
        r1, x1, y1 = pairs[1]

        if r0 < r1:
            alpha_star = opt_alpha(x0, y0, x1, y1, cx, cy, radius)
            xy_inter_point = bary_interpolation(x0, y0, x1, y1)(alpha_star)
        else:
            alpha_star = opt_alpha(x1, y1, x0, y0, cx, cy, radius)
            xy_inter_point = bary_interpolation(x1, y1, x0, y0)(alpha_star)

        inter_points.append(np.hstack((radius, xy_inter_point)))
    return np.array(inter_points)


def _append_polar_angle(
    edge_candidates: PlumePoints, orig_center: tuple[float, float]
) -> set[tuple[float, float, float, float]]:
    """
    Appends angle from orig_center based on (x,y) position.
    Branch cut `theta in [-pi, pi]`

    Returns:
    -------
    np.ndarray([[ri,xi,yi,ti],...])

    """
    cx, cy = orig_center

    polar_points = []
    for rad, x_pos, y_pos in edge_candidates:
        dy = y_pos - cy
        dx = x_pos - cx
        theta = np.arctan2(dy, dx)

        polar_points.append((rad, x_pos, y_pos, theta))

    return np.array(polar_points)


def _apply_concentric_search(
    img_gray: GrayImage,
    num_of_circs: int,
    orig_center: tuple[float, float],
    radii: int,
    interior_scale: float,
    rtol: float,
    atol: float,
    quiet: bool,
) -> PlumePoints:
    # Initialize numpy array to store center
    points_mean = np.zeros(shape=(num_of_circs + 1, 3))
    zero_index = 0
    val = 0
    points_mean[0] = np.insert(orig_center, zero_index, val)

    # Plot first point on path
    _, center = _find_max_on_circle(img_gray, orig_center, radius=radii)
    # code to choose only one

    points_mean[1] = np.insert(center, zero_index, radii)

    for step in range(2, num_of_circs + 1):
        radius = radii * step

        # Get center of next point
        error_occured = False
        try:
            _, center = _find_next_center(
                array=img_gray,
                orig_center=orig_center,
                neig_center=center,
                r=radius,
                scale=interior_scale,
                rtol=rtol,
                atol=atol,
            )

        except Exception as e:
            if quiet:
                print(f"Error occurred: {e}")
            error_occured = True

        if error_occured:
            break

        points_mean[step] = np.insert(center, zero_index, radius)
    return points_mean


def _find_max_on_circle(
    array: GrayImage, center: tuple[float, float], radius: float, n_points: int = 101
):
    """
    Find the max value (and index) of an `array` on a circle centered
    at `center` with `radius`. Values restricted to those that fall within
    array.

    Coordinates for circle calculated along linspace for horizontal axis.

    Parameters:
    ----------
    array:
        Array of float values.

    center:
        coordinate specifying center of circle

    radius:
        radius of circle.

    n_points:
        number of points used in linspace for calculating coordinates points
        of circle. Values must be sufficiently high to cover all points
        on circle. Safe minimum is 2/3*radius in pixel distance, rounded.

    Returns:
    -------
    max_value:
        max value along circle on array.

    max_indices:
        indices of max_value on array.
    """
    col, row = center
    height, width = array.shape

    def _get_x_values(col, r, n_points):
        x0 = np.max([0, col - r])
        x1 = np.min([width - 0.5, col + r])
        return np.linspace(x0, x1, n_points)

    def _y_in_range(yi, height):
        if round(yi) < 0:
            return False
        if round(yi) > height - 1:
            return False
        return True

    def _find_max_val_and_idx(arr):
        max_value = np.max(arr)
        max_indices = np.where(arr == max_value)[0]

        return max_value, max_indices

    xy_circle = []
    for xi in _get_x_values(col, radius, n_points):
        y0 = np.sqrt(radius**2 - (xi - col) ** 2) + row
        y1 = -np.sqrt(radius**2 - (xi - col) ** 2) + row

        if _y_in_range(y0, height):
            xy_circle.append((xi, y0))

        if _y_in_range(y1, height):
            xy_circle.append((xi, y1))

    xy_circle = np.unique(np.array(xy_circle).round().astype(int), axis=0)

    xy_circle_vals = np.array([array[y, x] for (x, y) in xy_circle])

    max_val, max_idx = _find_max_val_and_idx(xy_circle_vals)

    max_indices = xy_circle[max_idx].reshape(-1)

    return max_val, max_indices


def _find_next_center(
    array, orig_center, neig_center, r, scale=3 / 5, rtol=1e-3, atol=1e-6
):
    # print("entered find_next_center")
    col, row = orig_center
    n, d = array.shape

    # generate grid of indices from array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Get array of distances
    distances = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)

    # Create mask for points on boundary (distances == r)
    boundary_mask = np.isclose(distances, r, rtol=rtol, atol=atol)

    # Appply to neighboring point
    col, row = neig_center

    # get array of new distances from previous circle
    distances = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)

    interior_mask = distances <= r * scale

    search_mask = boundary_mask & interior_mask

    search_subset = array[search_mask]

    max_value = np.max(search_subset)

    # find indices of max element
    max_indices = np.argwhere(np.isclose(array, max_value) & search_mask)

    row, col = max_indices[0]

    max_indices = (col, row)

    return max_value, max_indices


def _contour_distances(
    contours: Contour_List, origin: tuple[float, float]
) -> list[PlumePoints]:
    """
    Gets L2 distances between array of contours and single point origin.
    """
    contour_dist_list = []
    for contour in contours:
        distances = np.sqrt(np.sum((contour - np.array(origin)) ** 2, axis=1))
        contour_dist = np.hstack((distances.reshape((-1, 1)), contour))
        contour_dist_list.append(contour_dist)

    return contour_dist_list


def _sol_in_contours(sols: Float2D, selected_contours: Contour_List) -> Bool1D:
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
