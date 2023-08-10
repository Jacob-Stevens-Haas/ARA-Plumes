# Script for testing variance detection through the use of cv2.findContours

import cv2
import matplotlib.pyplot as plt

# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0287.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0102.png"
img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0690.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0628.png"

#########################
## Ask user for center ##
#########################

# Global variable to store the clicked point
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at ({x}, {y})")
        cv2.destroyAllWindows()

# Load image 
image = cv2.imread(img_path) 

# Convert to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a window and set the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Display the image
cv2.imshow("Image", image_gray)

# Keep the script running until a click event occurs
while clicked_point is None:
    cv2.waitKey(1)  # Check for a key press every 1 millisecond


# After a click event, continue with the rest of the script
print("Clicked point:", clicked_point)

center = clicked_point

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
n=3
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