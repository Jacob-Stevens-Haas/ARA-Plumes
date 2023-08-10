# Script for testing variance detection through the use of cv2.findContours

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes/')

from utils import ImagePointPicker, find_max_on_boundary, find_next_center

img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0287.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0102.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0690.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0628.png"

#########################
## Ask user for center ##
#########################

image_to_click = ImagePointPicker(img_path)
image_to_click.ask_user()
orig_center = image_to_click.clicked_point 


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

# Creating distances
contour_dist_list = []
for contour in selected_contours:
    a = contour.reshape(-1,2)
    b = np.array(orig_center)
    contour_dist = np.sqrt(np.sum((a-b)**2,axis=1))
    contour_dist_list.append(contour_dist)


# test = np.isclose(contour_dist,2*50, rtol=1e-2)
# intersection_points=selected_contours[0].reshape(-1,2)[test]
# print(intersection_points)
# print("Printing intersection points;")
# for point in intersection_points:
#     print(point)


# print(True in test)
# print(np.count_nonzero(test))
# print(True in (contour_dist <= 50.5))





    

    # for point in contour:
    #     print(point)

# print("selected contours size:", selected_contours[0].shape)

# Draw contours on the original image (or a copy of it)
contour_image_original = image.copy()
cv2.drawContours(contour_image_original, selected_contours, -1, (0, 255, 0), 2)
cv2.circle(contour_image_original, orig_center, 8, (255, 0, 0), -1)  # Draw red circle based on input


##############################
## Apply Concentric Circles ##
##############################
radii = 50
num_of_circs = 22

fit_poly = True
boundary_ring = True
interior_ring = False
scale = 3/5
rtol=1e-2
atol=1e-6
poly_deg = 2
x_less = 600
x_plus = 50
blue_color = (255,0,0)
red_color = (0,0,255)

# Instatiate numpy array to store centers
points = np.zeros(shape=(num_of_circs+1,2))
points[0] = orig_center

# Instantie list to store variance points
var_points = []

# Plot first point on path
_, center = find_max_on_boundary(image_gray, orig_center,radii, rtol=rtol,atol=atol)
points[1]=center

##### Check variance intersection
for i in range(len(selected_contours)):
    contour_dist = contour_dist_list[i]
    contour = selected_contours[i]

    dist_mask =np.isclose(contour_dist,radii, rtol=1e-2)
    intersection_points = contour.reshape(-1,2)[dist_mask]

    for point in intersection_points:
        var_points.append(list(point))


# for contour_dist in contour_dist_list:
#     dist_mask =np.isclose(contour_dist,radii, rtol=1e-2)
#     intersection_points=selected_contours[0].reshape(-1,2)[dist_mask]



# Draw rings
if boundary_ring == True:
    cv2.circle(contour_image_original, orig_center, radii, (0,0,255),1, lineType = cv2.LINE_AA)

for step in range(2, num_of_circs+1):
    radius = radii*step

    ##### Check variance intersection
    for i in range(len(selected_contours)):
        contour_dist = contour_dist_list[i]
        contour = selected_contours[i]

        dist_mask =np.isclose(contour_dist,radius, rtol=1e-2)
        intersection_points = contour.reshape(-1,2)[dist_mask]

    for point in intersection_points:
        var_points.append(list(point))
    

    # Draw interior ring
    if interior_ring == True:
        cv2.circle(contour_image_original,
                   center,
                   int(radius*scale),
                   blue_color,
                   1,
                   lineType=cv2.LINE_AA)
        
    
    # Get center of next point
    ## Throw in try catch error --> break out of loop if does not work.
    error_occured = False
    try:
        _, center = find_next_center(array=image_gray,
                                    orig_center=orig_center,
                                    neig_center=center,
                                    r=radius,
                                    scale=scale,
                                    rtol=rtol,
                                    atol=atol)
    except Exception as e:
        print("empty search")
        error_occured = True
    
    if error_occured == True:
        break   
    
    points[step] = center

    # Draw next point
    cv2.circle(contour_image_original, center,5,red_color,-1)

    # Draw boundary ring
    if boundary_ring == True:
        cv2.circle(contour_image_original,
                   center=orig_center,
                   radius=radius,
                   color=red_color,
                   thickness=1,
                   lineType=cv2.LINE_AA)

for point in var_points:
    cv2.circle(contour_image_original, point, 5, blue_color,-1)

####################
## Apply Poly fit ##
####################
poly_coef = np.polyfit(points[:,0], points[:,1],deg=poly_deg)

x = np.linspace(np.min(points[:,0])-x_less,np.max(points[:,0])+x_plus,100)
y = poly_coef[0]*x**2 + poly_coef[1]*x+poly_coef[2]

curve_img = np.zeros_like(contour_image_original)
curve_points = np.column_stack((x,y)).astype(np.int32)

cv2.polylines(curve_img, [curve_points], isClosed=False,color=blue_color,thickness=5)
if fit_poly==True:
    contour_image_original = cv2.addWeighted(contour_image_original,1,curve_img,1,0)



#################################
## Apply smoothing to Contours ##
#################################

# Create an empty list to store smoothed contours
smoothed_contours = []

# Apply contour smoothing (contour approximation) to each selected contour
epsilon = 100  # Adjust epsilon as needed
for contour in selected_contours:
    smoothed_contours.append(cv2.approxPolyDP(contour, epsilon, True))

# print("Smoothed contours size:", smoothed_contours[0].shape)
# Draw smoothed contours on the original image (or a copy of it)
contour_image_smoothed = image.copy()
cv2.drawContours(contour_image_smoothed, smoothed_contours, -1, (0, 255, 0), 2)
cv2.circle(contour_image_smoothed, orig_center, 5, (255, 0, 0), -1)  # Draw red circle based on input

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