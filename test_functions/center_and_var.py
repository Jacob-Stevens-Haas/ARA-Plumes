import cv2
import matplotlib.pyplot as plt

img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/July_20/video_low_1/fixed_avg_frames/subtract_0287.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/July_20/video_low_1/fixed_avg_frames/subtract_0107.png"

# Load image 
image = cv2.imread(img_path,0) # Load in as grayscale

########################
## Apply Thresholding ##
########################

# Simple Thresholding
# threshold_value = 60
# _, threshold = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# OTSU thresholding (automatically choose params)
_, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Thresholding
# threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

############################
## Find Contours in image ##
############################

# Find Contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select n largest contours
n=3
contours.sort(key=lambda c: cv2.contourArea(c), reverse = True)

selected_contours = contours[:n]

# Draw contours on the original image (or a copy of it)
contour_image = image.copy()
cv2.drawContours(contour_image, selected_contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow("Image with Contours", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

###################################
## Get center points of contours ##
###################################

# Calculate center of mass
center_points = []

for contour in selected_contours:
    # Calculate moments of the contour
    moments = cv2.moments(contour)

    # Calculate center of mass
    if moments["m00"] !=0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        center_points.append((center_x, center_y))

##############################
## Plot centers of contours ##
##############################

# Convert the OpenCV image to a format compatible with matplotlib
plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a scatter plot for the center points
plt.figure(figsize=(8, 6))
plt.imshow(plt_image)
plt.scatter(*zip(*center_points), color='red', marker='x', s=50)
plt.title("Plume with Center Points")
plt.axis('off')
plt.show()

# print(type(contours))
# print(center_points)
# print("Number of contours:", len(contours)) 