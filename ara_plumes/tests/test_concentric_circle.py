import numpy as np

from ..concentric_circle import _append_polar_angle
from ..concentric_circle import _contour_distances
from ..concentric_circle import _find_intersections
from ..concentric_circle import _find_max_on_circle
from ..concentric_circle import _get_edge_points
from ..concentric_circle import _interpolate_intersections
from ..concentric_circle import concentric_circle
from ..models import get_contour
from ..typing import X_pos
from ..typing import Y_pos


def test_concentric_circle():
    # create data
    scale = 4
    orig_center = (X_pos(50), Y_pos(200))
    height, width = 400, 400
    bw_img = np.zeros((height, width), dtype=np.uint8)

    bw_img[25:376, 50] = 255 // scale
    bw_img[25, 50:251] = 255 / scale
    bw_img[375, 50:251] = 255 // scale
    bw_img[25:376, 250] = 255 // scale
    bw_img[200, [100, 150, 200]] = 255
    rectangle_plume = bw_img

    num_of_circs = 3
    r = 50
    selected_contours = get_contour(rectangle_plume)
    result_mean, result_var1, result_var2 = concentric_circle(
        rectangle_plume,
        contours=selected_contours,
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


def test_find_max_on_circle():
    axis = np.arange(5).reshape(-1, 1)
    # arr is surface u = x*y in positive quadrant
    arr = axis @ axis.T
    radii = [x_dist * np.sqrt(2) for x_dist in axis.flatten()]
    for x_dist, radius in zip(axis, radii):
        max_val, amax = _find_max_on_circle(arr, (0, 0), radius)
        # Max should be on the diagonal, i.e. x = y
        assert max_val == x_dist[0] ** 2
        np.testing.assert_array_equal(amax, [x_dist[0], x_dist[0]])


def test_get_edge_points():
    # Image of test
    # https://www.desmos.com/calculator/umi82ahhey
    radii = 1
    contour_distances = [
        np.array(
            [
                [np.sqrt(2), 0, 0],
                [np.sqrt(2), 2, 0],
                [np.sqrt(10), 2, 2],
                [np.sqrt(10), 0, 2],
            ]
        )
    ]
    orig_center = (1, -1)
    ccw, cw = _get_edge_points(1, radii, contour_distances, orig_center)

    # test empty array when no intersection is found.
    assert len(ccw) == 0
    assert len(cw) == 0

    ccw, cw = _get_edge_points(3, radii, contour_distances, orig_center)

    # test ccw points have greater angle than cw points.
    for ccw_point, cw_point in zip(ccw, cw):
        ccw_theta = np.arctan2(ccw_point[2], ccw_point[1])
        cw_theta = np.arctan2(cw_point[2], cw_point[1])

        assert ccw_theta >= cw_theta


def test_contour_distances():
    contour = np.array([[0, 0], [0, 1]])
    origin = (1.0, 0.0)
    ranges = np.array([[1], [np.sqrt(2)]])
    expected = [np.hstack((ranges, contour))]
    result = _contour_distances([contour], origin=origin)
    np.testing.assert_allclose(result, expected)


def test_append_polar_angle():
    edge_candidates = np.array(
        [[1, 1 / 2, np.sqrt(3) / 2], [1, -1 / 2, np.sqrt(3) / 2]]
    )
    orig_center = (0, 0)

    result = _append_polar_angle(edge_candidates, orig_center)
    expected = np.array(
        [
            [1, 1 / 2, np.sqrt(3) / 2, np.pi / 3],
            [1, -1 / 2, np.sqrt(3) / 2, 2 * np.pi / 3],
        ]
    )

    np.testing.assert_array_almost_equal(expected, result)


def test_interpolate_intersections():
    contour_crosses = [np.array([[0, 0, 0], [2 * np.sqrt(2), 2, 2]])]
    radius = np.sqrt(2)
    orig_center = (0, 0)
    result = _interpolate_intersections(contour_crosses, radius, orig_center)
    dist_to_origin = np.linalg.norm(result[0, 1:] - orig_center)

    # test distance from result to orig_center
    np.testing.assert_equal(dist_to_origin, radius)

    x0, y0 = contour_crosses[0][0, 1:]
    x1, y1 = contour_crosses[0][1, 1:]
    x_result, y_result = result[0, 1:]

    slope1 = (y_result - y0) / (x_result - x0)
    slope2 = (y1 - y_result) / (x1 - x_result)

    np.testing.assert_equal(slope1, slope2)


def test_find_intersections():
    contours = [
        np.array(
            [
                [np.sqrt(2), 0, 0],
                [np.sqrt(2), 2, 0],
                [np.sqrt(10), 2, 2],
                [np.sqrt(10), 0, 2],
            ]
        )
    ]

    expected = [
        np.array([[np.sqrt(10), 0, 2], [np.sqrt(2), 0, 0]]),
        np.array([[np.sqrt(2), 2, 0], [np.sqrt(10), 2, 2]]),
    ]
    result = _find_intersections(contours, radius=2)

    for r, e in zip(result, expected):
        np.testing.assert_array_almost_equal(r, e)


def test_find_intersections_2_contours():
    contours_2 = [
        np.array(
            [
                [np.sqrt(2), 0, 0],
                [np.sqrt(2), 2, 0],
                [np.sqrt(10), 2, 2],
                [np.sqrt(10), 0, 2],
            ]
        ),
        np.array(
            [
                [np.sqrt(2), 2, 0],
                [np.sqrt(2), 2, -2],
                [np.sqrt(10), 4, -2],
                [np.sqrt(10), 4, 0],
            ]
        ),
    ]

    expected = [
        np.array([[np.sqrt(10), 0, 2], [np.sqrt(2), 0, 0]]),
        np.array([[np.sqrt(2), 2, 0], [np.sqrt(10), 2, 2]]),
        np.array([[np.sqrt(10), 4, 0], [np.sqrt(2), 2, 0]]),
        np.array([[np.sqrt(2), 2, -2], [np.sqrt(10), 4, -2]]),
    ]
    result = _find_intersections(contours_2, radius=2)

    for r, e in zip(result, expected):
        np.testing.assert_array_almost_equal(r, e)
