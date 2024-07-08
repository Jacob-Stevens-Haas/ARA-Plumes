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
    poly_deg: float = 2,
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

    poly_deg:
        Specifying degree of polynomail to learn on points selected from
        concentric circle method.

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
    points_mean = cast(PlumePoints, np.vstack((points_mean[:0], points_mean[1:][mask])))

    # points_mean[:, 1:] -= orig_center

    # # Checking edge points #
    # poly_coef_mean = np.polyfit(points_mean[:, 1], points_mean[:, 2], deg=poly_deg)

    # f_mean = (
    #     lambda x: poly_coef_mean[0] * x**2 + poly_coef_mean[1] * x + poly_coef_mean[2]
    # )

    contour_dist_list = _contour_distances(contours, orig_center)

    # Initialize edge list to store points
    points_var1, points_var2 = _get_var_points(
        num_of_circs, radii, contours, contour_dist_list, orig_center, f_mean
    )

    # add origin center back
    points_mean[:, 1:] += orig_center
    points_var1[:, 1:] += orig_center
    points_var2[:, 1:] += orig_center

    return points_mean, points_var1, points_var2


def _get_var_points(
    num_of_circs, radii, selected_contours, contour_dist_list, orig_center, f_mean
):
    zero_index = 0
    # Initialize variance list to store points
    var1_points = []
    var2_points = []

    for step in range(1, num_of_circs + 1):
        radius = step * radii

        var_above = []
        var_below = []

        intersection_points = []
        for i in range(len(selected_contours)):
            contour_dist = contour_dist_list[i]
            contour = selected_contours[i]

            # ISSUE PROBABLY IS HERE
            dist_mask = np.isclose(contour_dist, radius, rtol=1e-2)
            intersection_points_i = contour[dist_mask]
            intersection_points.append(intersection_points_i)

        # TO DO: re translate these - DONE
        for ip_i in intersection_points:
            for point in ip_i:
                if f_mean(point[0] - orig_center[0]) <= point[1] - orig_center[1]:
                    var_above.append(point - orig_center)
                else:
                    var_below.append(point - orig_center)

        if bool(var_above):
            # Average the selected variance points (if multiple selected)
            avg_var1_i = np.array(var_above).mean(axis=0).round().astype(int)
            # Insert associated radii
            avg_var1_i = np.insert(avg_var1_i, zero_index, radius)
            var1_points.append(list(avg_var1_i))

        if bool(var_below):
            # Average the selected variance points (if multiple selected)
            avg_var2_i = np.array(var_below).mean(axis=0).round().astype(int)
            # Insert associated radii
            avg_var2_i = np.insert(avg_var2_i, zero_index, radius)
            var2_points.append(list(avg_var2_i))

    def _insert_origin(vari_points):
        if vari_points:
            return np.vstack((list(np.insert((0, 0), 0, 0)), np.array(vari_points)))
        return np.array([[0, 0, 0]])

    points_var1 = _insert_origin(var1_points)
    points_var2 = _insert_origin(var2_points)

    return points_var1, points_var2


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
    _, center = _find_max_on_boundary(img_gray, orig_center, r=radii)
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


def _find_max_on_boundary(array, center, r, n_points=101):
    """
    Find the max value (and index) of an array on a circle centered
    at center with radii r

    Parameters:
    ----------
    """
    col, row = center
    n, d = array.shape

    def _get_x_values(col, r, n_points):
        x0 = np.max([0, col - r])
        x1 = np.min([d, col + r])
        return np.linspace(x0, x1, n_points)

    def _y_in_range(yi, row):
        if round(yi) < 0:
            return False
        if round(yi) > row:
            return False
        return True

    def _find_max_val_and_idx(arr):
        max_value = np.max(arr)
        max_indices = np.where(arr == max_value)[0]

        return max_value, max_indices

    xy_circle = []
    for xi in _get_x_values(col, r, n_points):
        y0 = np.sqrt(r**2 - (xi - col) ** 2) + row
        y1 = -np.sqrt(r**2 - (xi - col) ** 2) + row

        if _y_in_range(y0, row):
            xy_circle.append((xi, y0))

        if _y_in_range(y1, row):
            xy_circle.append((xi, y1))

    xy_circle = np.array(xy_circle).round().astype(int)

    xy_circle_vals = np.array([array[y, x] for (x, y) in xy_circle])

    max_val, max_idx = _find_max_val_and_idx(xy_circle_vals)

    max_indices = np.unique(xy_circle[max_idx], axis=0).reshape(-1)  # something funky

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
