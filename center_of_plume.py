import cv2
import numpy as np

img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixed_avg_frames/subtract_10465.png"
img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display original image
# cv2.imshow("Input frame", img)

# Show gray image
# cv2.imshow("Gray image", gray)


# Plot a red dot at (x,y)
x = 170
y = 135

red_color = (0,0,255)

cv2.circle(img,
            (x,y),
            3,
            red_color,
            -1)

# plot blue dot at the new (x,y)
x = 218
y = 135

blue_color = (255,0,0)

radius = 3

# cv2.circle(img, (x,y), radius, blue_color,-1)

# Show image with red dot 
cv2.imshow("image with dots", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

# row, col = gray.shape

# for i in range(col):
#     print(gray[40,:][i])

# print(type(gray))
# print("max element:", max(gray))

###############################################################################

def find_max_on_boundary(array, x, y, r):
    n, d = array.shape

    # Generate a grid of indices for the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Calculate the Euclidean distance from each point to the center
    distances = np.sqrt((xx - x)**2 + (yy - y)**2)

    # Create a mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances, r)

    # Apply the boundary mask to the array to get the subset
    boundary_subset = array[boundary_mask]

    # Find the maximum value within the subset
    max_value = np.max(boundary_subset)

    # Find the indices of the maximum elements within the boundary
    max_indices = np.argwhere(np.isclose(array, max_value) & boundary_mask)

    # Return the maximum value and its locations
    return max_value, max_indices


################################################################################


def find_max_on_boundary(array, x, y, r):
    n, d = array.shape

    # Generate a grid of indices for the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Calculate the Euclidean distance from each point to the center
    # calculate only once, once we have a center
    distances = np.sqrt((xx - x)**2 + (yy - y)**2)

    # Create a mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances, r)

    # Apply the boundary mask to the array to get the subset
    boundary_subset = array[boundary_mask]

    # Find the maximum value within the subset
    max_value = np.max(boundary_subset)

    # Find the indices of the maximum element within the boundary
    max_indices = np.argwhere(array == max_value)

    # Return the maximum value and its location
    return max_value, max_indices
