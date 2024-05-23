import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .typing import ColorImage
from .typing import Contour_List
from .typing import GrayImage
from .typing import Optional
from .typing import PointsMean
from .typing import PointsVar1
from .typing import PointsVar2
from .typing import X_pos
from .typing import Y_pos


def concentric_circle(
    img: GrayImage | ColorImage,
    selected_contours: Contour_List,
    orig_center: tuple[int, int],
    radii: int = 50,
    num_of_circs: int = 30,
    interior_scale: float = 3 / 5,
    rtol: float = 1e-2,
    atol: float = 1e-6,
    poly_deg: float = 2,
    mean_smoothing: bool = True,
    mean_smoothing_sigma: int = 2,
    quiet: bool = True,
) -> tuple[PointsMean, PointsVar1, PointsVar2]:
    """
    Applies concentric cirlces to a single frame (gray or BGR) from video

    Creates new image and learned poly coef for mean line and variance line.

    Parameters:
    -----------
    img: np.ndarray (gray or BGR)
        image to apply concentric circle too.

    selected_contours: list
        List of contours learned from get_contour

    orig_center:
        (x,y) coordinates of plume leak source.

    radii: int (default 50)
        The radii used to step out in concentric circles method.

    num_of_circles: int, optional (default 22)
        number of circles and radii steps to take in concentric circles method.

    interior_scale: float (default 3/5)
        Used to scale down the radii used on the focusing rings. Called in
        find_next_center

    rtol, atol: float (default 1e-2, 1e-6)
        Relative and absolute tolerances. Used in np.isclose function in
        find_max_on_boundary and find_next_center functions.
        Checks if points are close to selected radii.

    poly_deg: int (default 2)
        Specifying degree of polynomail to learn on points selected from
        concentric circle method.

    mean_smoothing: bool (default True)
        Applying additional gaussian filter to learned concentric circle
        points. Only in y direction

    mean_smoothing_sigma: int (default 2)
        Sigma parameter to be passed into gaussian_filter function when
        ``mean_smoothing = True``.

    quiet: bool (default True)
        suppresses error output


    Returns:
    --------
    points_mean:
        Returns nx3 array containing observed points along mean path.
        Where the kth entry is of the form [r(k), x(k), y(k)], i.e the
        coordinate (x,y) of the highest value point along the concetric circle
        with radii r(k).
        Note: (x,y) coordinates are re-centered to origin (0,0).

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

    # convert image to gray
    if len(img.shape) == 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    ############################
    # Apply Concentric Circles #
    ############################

    # array of distance of each point on contour to original center
    contour_dist_list = []
    for contour in selected_contours:
        a = contour.reshape(-1, 2)
        b = np.array(orig_center)
        contour_dist = np.sqrt(np.sum((a - b) ** 2, axis=1))
        contour_dist_list.append(contour_dist)

    # Initialize numpy array to store center
    points_mean = np.zeros(shape=(num_of_circs + 1, 3))
    zero_index = 0
    val = 0
    points_mean[0] = np.insert(orig_center, zero_index, val)

    # Plot first point on path
    _, center = _find_max_on_boundary(
        img_gray, orig_center, r=radii, rtol=rtol, atol=atol
    )

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

    ##########################
    # Apply poly fit to mean #
    ##########################

    # Apply gaussian filtering to points in y direction
    if mean_smoothing:
        smooth_x = points_mean[:, 1]
        smooth_y = gaussian_filter(points_mean[:, 2], sigma=mean_smoothing_sigma)
        points_mean[:, 1:] = np.column_stack((smooth_x, smooth_y))

    #########################################
    # Check if points fall within a contour #
    #########################################

    new_points_mean = []
    for center in points_mean:
        for contour in selected_contours:
            # check if point lies within contour
            if cv2.pointPolygonTest(contour, center[1:], False) == 1:
                new_points_mean.append(center)
                center[1:] = center[1:].round().astype(int)

    if bool(new_points_mean):
        points_mean = np.array(new_points_mean).reshape(-1, 3)

    points_mean[:, 1:] -= orig_center

    poly_coef_mean = np.polyfit(points_mean[:, 1], points_mean[:, 2], deg=poly_deg)

    f_mean = (
        lambda x: poly_coef_mean[0] * x**2 + poly_coef_mean[1] * x + poly_coef_mean[2]
    )

    ############################
    # Checking Variance points #
    ############################

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
            intersection_points_i = contour.reshape(-1, 2)[dist_mask]
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
            avg_var1_i = np.insert(avg_var1_i, 0, radius)
            var1_points.append(list(avg_var1_i))

        if bool(var_below):
            # Average the selected variance points (if multiple selected)
            avg_var2_i = np.array(var_below).mean(axis=0).round().astype(int)
            # Insert associated radii
            avg_var2_i = np.insert(avg_var2_i, 0, radius)
            var2_points.append(list(avg_var2_i))

    # Concatenate original center to both lists
    # TO DO: concatenate (0,0) to each list - DONE
    if bool(var1_points):
        points_var1 = np.vstack((np.array(var1_points), list(np.insert((0, 0), 0, 0))))
    else:
        points_var1 = np.insert((0, 0), 0, 0).reshape(1, -1)

    if bool(var2_points):
        points_var2 = np.vstack((np.array(var2_points), list(np.insert((0, 0), 0, 0))))
    else:
        points_var2 = np.insert((0, 0), 0, 0).reshape(1, -1)

    points_mean += orig_center
    points_var1 += orig_center
    points_var2 += orig_center

    return points_mean, points_var1, points_var2


def _find_max_on_boundary(array, center, r, rtol=1e-3, atol=1e-6):
    col, row = center
    n, d = array.shape

    # Generate a grid of indices for the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Get Euclidean distance from each point on grid to center
    distances = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)

    # Create mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances, r, rtol=rtol, atol=atol)

    # Apply the boundary mask to the array to get the subset
    boundary_subset = array[boundary_mask]

    # Find the maximum value within the subset
    max_value = np.max(boundary_subset)

    # Find the indices of the maximum elements within the boundary
    max_indices = np.argwhere(np.isclose(array, max_value) & boundary_mask)

    row, col = max_indices[0]

    max_indices = (col, row)

    return max_value, max_indices


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


def _add_contours_on_img(
    img: GrayImage | ColorImage,
    orig_center: tuple[int, int],
    mean_scatter: Optional[np.ndarray[tuple[X_pos, Y_pos]]] = None,
    var1_scatter: Optional[np.ndarray[tuple[X_pos, Y_pos]]] = None,
    var2_scatter: Optional[np.ndarray[tuple[X_pos, Y_pos]]] = None,
    selected_contours: Contour_List = None,
    radii: Optional[int] = None,
    num_of_circs: Optional[int] = None,
    interior_scale: Optional[float] = None,
    scatter_color=(0, 0, 255),
    ring_color=(255, 0, 0),
    interior_ring_color=(255, 0, 0),
    contour_color=(0, 255, 0),
) -> ColorImage:
    """
    Apply optional contour plotting to img.

    Returns:
    -------
    color_img:
        Colored frame with contour plotting applied.
    """

    if len(img.shape) == 2:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        color_img = img.copy()

    if mean_scatter is not None:
        for x_y in mean_scatter:
            cv2.circle(color_img, x_y, 7, scatter_color, -1)

    if var1_scatter is not None:
        for x_y in var1_scatter:
            cv2.circle(color_img, x_y, 7, scatter_color, -1)

    if var2_scatter is not None:
        for x_y in var2_scatter:
            cv2.circle(color_img, x_y, 7, scatter_color, -1)

    if selected_contours:
        cv2.drawContours(color_img, selected_contours, -1, contour_color, 2)

    if radii and num_of_circs:
        for step in range(1, num_of_circs + 1):
            radius_i = radii * step

            cv2.circle(
                color_img,
                center=orig_center,
                radius=radius_i,
                color=ring_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    if interior_scale and mean_scatter is not None:
        for i, point in enumerate(mean_scatter[1:]):
            i += 2
            radius_i = radii * i
            cv2.circle(
                color_img,
                center=point,
                radius=int(radius_i * interior_scale),
                color=interior_ring_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return color_img
