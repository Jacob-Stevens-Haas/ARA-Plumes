import cv2
import numpy as np

img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixed_avg_frames/subtract_16014.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixedd_avg_frames/subtract_10465.png"
img1 = cv2.imread(img_path)

# # convert to gray
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# # Using the Canny filter to get contours
# edges = cv2.Canny(gray, 10, 30)
# # Using the Canny filter with different parameters
# edges_high_thresh = cv2.Canny(gray, 5, 20)
# # Stacking the images to print them together
# # For comparison
# images = np.hstack((gray, edges, edges_high_thresh))

# # Display the resulting frame
# cv2.imshow('Frame', images)

# # waiting until key press
# cv2.waitKey()

# # destroy all the windows
# cv2.destroyAllWindows()

############################################################

# convert to gray
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Smoothing without removing edges.
gray_filtered = cv2.bilateralFilter(gray, 10, 20, 20)

gray_filtered_2 = cv2.bilateralFilter(gray, 5,5,10)

# Using the Canny filter to get contours
edges_filtered = cv2.Canny(gray_filtered, 5, 10)
edges_filtered_2 = cv2.Canny(gray_filtered_2, 5,10)

# edges_filtered = cv2.Canny(gray_filtered, threshold1=5, threshold2=10)
# edges_filtered_2 = cv2.Canny(gray_filtered_2, threshold1=5,threshold2=10)

# Stacking the images to print them together
# For comparison
images = np.hstack((gray, edges_filtered, edges_filtered_2))

# Display the resulting frame
cv2.imshow('Frame', images)

# waiting until key press
cv2.waitKey(0)

cv2.destroyAllWindows()

############################################################

# Check different hyperparameter values


# Define BilateralFilter values
values = range(10,90,10)
# d_vals = values
d = 25 # Or maybe even 20
sigmaColor = 20
sigmaSpace = 30

# Convert image into gray 
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

imgs = [gray]

for sigmaSpace in values:
    # Apply bilateral filtering
    gray_filtered = cv2.bilateralFilter(gray,d,sigmaColor,sigmaSpace)

    # Detect edges
    edges_filtered = cv2.Canny(gray_filtered, 5,10)

    imgs.append(edges_filtered)


def show_images_3x3(images):
    frame = np.ones((1220, 1220), dtype=np.uint8) * 255  # Create a white frame

    for i, image in enumerate(images):
        row = i // 3
        col = i % 3
        resized_image = cv2.resize(image, (400, 400))

        # Add white borders to the resized image with a border size of 2 pixels
        bordered_image = cv2.copyMakeBorder(
            resized_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        frame[row * 408: (row * 408) + 404, col * 408: (col * 408) + 404] = bordered_image

    cv2.imshow('3x3 Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show_images_3x3(imgs)