import numpy as np
from ..concentric_circle import concentric_circle
from ..models import get_contour

def test_concentric_circle():
    # create data
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

    # get result
    selected_contours = get_contour(rectangle_plume)
    result_mean, result_var1, result_var2 = concentric_circle(
        rectangle_plume,
        selected_contours=selected_contours,
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

