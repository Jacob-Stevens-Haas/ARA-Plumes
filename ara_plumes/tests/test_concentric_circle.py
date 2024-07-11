import numpy as np

from ..concentric_circle import _add_polar_angle
from ..concentric_circle import _contour_distances
from ..concentric_circle import _find_intersections
from ..concentric_circle import _find_max_on_boundary
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

    num_of_circs = 3
    r = 50
    for i in range(num_of_circs):
        i += 2
        bw_img[200, r * i] = 255

    rectangle_plume = bw_img.copy()

    # get result
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


def test_find_max_on_boundary():
    scale = 4
    orig_center = (50, 200)
    height, width = 400, 400
    bw_img = np.zeros((height, width), dtype=np.uint8)

    bw_img[25:376, 50] = 255 // scale
    bw_img[25, 50:251] = 255 / scale
    bw_img[375, 50:251] = 255 // scale
    bw_img[25:376, 250] = 255 // scale

    num_of_circs = 3
    r = 50
    for i in range(num_of_circs):
        i += 2
        bw_img[200, r * i] = 255

    rectangle_plume = bw_img.copy()

    expected = (255, np.array([100, 200]))
    result = _find_max_on_boundary(rectangle_plume, orig_center, r=50)

    np.testing.assert_equal(expected[0], result[0])
    np.testing.assert_array_almost_equal(expected[1], result[1])


def test_get_edge_points_3_circles():
    num_of_circs = 3
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
    result = _get_edge_points(num_of_circs, radii, contour_distances, orig_center)
    expected = (
        np.array([[2, 0, np.sqrt(3) - 1], [3, 0, 2 * np.sqrt(2) - 1]]),
        np.array([[2, 2, np.sqrt(3) - 1], [3, 2, 2 * np.sqrt(2) - 1]]),
    )
    for e, r in zip(expected, result):
        np.testing.assert_array_almost_equal(e, r)


def test_contour_distances():
    contour = np.array([[0, 0], [0, 1]])
    origin = (1.0, 0.0)
    ranges = np.array([[1], [np.sqrt(2)]])
    expected = [np.hstack((ranges, contour))]
    result = _contour_distances([contour], origin=origin)
    np.testing.assert_allclose(result, expected)


def test__add_polar_angle():
    edge_candidates = np.array(
        [[1, 1 / 2, np.sqrt(3) / 2], [1, -1 / 2, np.sqrt(3) / 2]]
    )
    orig_center = (0, 0)

    result = _add_polar_angle(edge_candidates, orig_center)
    expected = np.array(
        [
            [1, 1 / 2, np.sqrt(3) / 2, np.pi / 3],
            [1, -1 / 2, np.sqrt(3) / 2, 2 * np.pi / 3],
        ]
    )

    np.testing.assert_array_almost_equal(expected, result)


def test_interpolate_intersections():
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
    orig_center = (1, -1)
    radius = 2
    contour_crosses = _find_intersections(contours, radius)
    result = _interpolate_intersections(contour_crosses, radius, orig_center)

    expected = np.array([[2, 0, np.sqrt(3) - 1], [2, 2, np.sqrt(3) - 1]])

    np.testing.assert_array_almost_equal(expected, result)


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
