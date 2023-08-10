# Script for testing variance detection through the use of cv2.findContours

import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes/')

from utils import ImagePointPicker

# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0287.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0102.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0690.png"
img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0628.png"

#########################
## Ask user for center ##
#########################

image_to_click = ImagePointPicker(img_path)
image_to_click.ask_user()
center = image_to_click.clicked_point 

print("test")
print(center)

# Load image and convert to gray
image = cv2.imread(img_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


########################
## Apply Thresholding ##
########################

# Simple Thresholding
# threshold_value = 60
# _, threshold = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

# OTSU thresholding (automatically choose params)
_, threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

## Adaptive Thresholding
# threshold = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

############################
## Find Contours in image ##
############################

# Find Contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Select n largest contours
n=1
contours = sorted(contours,key=cv2.contourArea, reverse = True)
selected_contours = contours[:n]

# Draw contours on the original image (or a copy of it)
contour_image_original = image.copy()
cv2.drawContours(contour_image_original, selected_contours, -1, (0, 255, 0), 2)
cv2.circle(contour_image_original, center, 5, (0, 0, 255), -1)  # Draw red circle based on input

#################################
## Apply smoothing to Contours ##
#################################

# Create an empty list to store smoothed contours
smoothed_contours = []

# Apply contour smoothing (contour approximation) to each selected contour
epsilon = 125  # Adjust epsilon as needed
for contour in selected_contours:
    smoothed_contours.append(cv2.approxPolyDP(contour, epsilon, True))


# Draw smoothed contours on the original image (or a copy of it)
contour_image_smoothed = image.copy()
cv2.drawContours(contour_image_smoothed, smoothed_contours, -1, (0, 255, 0), 2)
cv2.circle(contour_image_smoothed, center, 5, (0, 0, 255), -1)  # Draw red circle based on input

# Display the images side by side
comparison_image = cv2.hconcat([contour_image_original, contour_image_smoothed])

# Display the image with contours
cv2.imshow(f"Orig vs eps={epsilon} smoothed Contours", comparison_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




###################################
## Get center points of contours ##
###################################

# # Calculate center of mass
# center_points = []

# for contour in selected_contours:
#     # Calculate moments of the contour
#     moments = cv2.moments(contour)

#     # Calculate center of mass
#     if moments["m00"] !=0:
#         center_x = int(moments["m10"] / moments["m00"])
#         center_y = int(moments["m01"] / moments["m00"])
#         center_points.append((center_x, center_y))

# ##############################
# ## Plot centers of contours ##
# ##############################

# # Convert the OpenCV image to a format compatible with matplotlib
# plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Create a scatter plot for the center points
# plt.figure(figsize=(8, 6))
# plt.imshow(plt_image)
# plt.scatter(*zip(*center_points), color='red', marker='x', s=50)
# plt.title("Plume with Center Points")
# plt.axis('off')
# plt.show()