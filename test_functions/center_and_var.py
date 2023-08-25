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

def learn_center_var(img_path,
                     orig_center,
                     smoothing = False,
                     eps_smooth = 50,
                     radii = 50,
                     num_of_circs = 22,
                     fit_poly = True,
                     boundary_ring = True,
                     interior_ring = False,
                     scale = 3/5,
                     rtol=1e-2,
                     atol=1e-6,
                     poly_deg=2,
                     x_less = 600,
                     x_plus = 0):
    
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

    #############################
    ## Apply Contour Smoothing ##
    #############################
    if smoothing == True:
        smoothed_contours = []

        for contour in selected_contours:
            smoothed_contours.append(cv2.approxPolyDP(contour, eps_smooth, True))
        
        selected_contours = smoothed_contours


    # Creating distances contours - array of distance of each point on contour to original center
    contour_dist_list = []
    for contour in selected_contours:
        a = contour.reshape(-1,2)
        b = np.array(orig_center)
        contour_dist = np.sqrt(np.sum((a-b)**2,axis=1))
        contour_dist_list.append(contour_dist)



    # Draw contours on the original image (or a copy of it)
    contour_image_original = image.copy()
    cv2.drawContours(contour_image_original, selected_contours, -1, (0, 255, 0), 2)
    cv2.circle(contour_image_original, orig_center, 8, (255, 0, 0), -1)  # Draw red circle based on input

    ##############################
    ## Apply Concentric Circles ##
    ##############################
    blue_color = (255,0,0)
    red_color = (0,0,255)

    # Instatiate numpy array to store centers
    points_mean = np.zeros(shape=(num_of_circs+1,2))
    points_mean[0] = orig_center

    # Instantiate list to store variance points
    var_points = []

    # Plot first point on path
    _, center = find_max_on_boundary(image_gray, orig_center,radii, rtol=rtol,atol=atol)
    points_mean[1]=center

    # Draw rings
    if boundary_ring == True:
        cv2.circle(contour_image_original, orig_center, radii, (0,0,255),1, lineType = cv2.LINE_AA)

    for step in range(2, num_of_circs+1):
        radius = radii*step

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
        
        points_mean[step] = center

        # Draw next point
        cv2.circle(contour_image_original, center,7,blue_color,-1)

        # Draw boundary ring
        if boundary_ring == True:
            cv2.circle(contour_image_original,
                    center=orig_center,
                    radius=radius,
                    color=red_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)
            
    ############################
    ## Apply Poly fit to mean ##
    ############################
    poly_coef_mean = np.polyfit(points_mean[:,0], points_mean[:,1],deg=poly_deg)

    f_mean = lambda x: poly_coef_mean[0]*x**2 + poly_coef_mean[1]*x+poly_coef_mean[2]

    x = np.linspace(np.min(points_mean[:,0])-x_less,np.max(points_mean[:,0])+x_plus,100)
    y = f_mean(x)

    curve_img = np.zeros_like(contour_image_original)
    curve_points = np.column_stack((x,y)).astype(np.int32)

    cv2.polylines(curve_img, [curve_points], isClosed=False,color=blue_color,thickness=5)
    if fit_poly==True:
        contour_image_original = cv2.addWeighted(contour_image_original,1,curve_img,1,0)
    
    ##############################
    ## Checking Variance points ##
    ##############################

    # Initialize variance list to store points
    var1_points = []
    var2_points = []

    for step in range(1, num_of_circs+1):
        radius = step*radii

        var_above = []
        var_below = []

        for i in range(len(selected_contours)):
            contour_dist = contour_dist_list[i]
            contour = selected_contours[i]

            dist_mask = np.isclose(contour_dist,radius, rtol=1e-2)
            intersection_points = contour.reshape(-1,2)[dist_mask]
        
        for point in intersection_points:
            if f_mean(point[0]) <= point[1]:
                var_above.append(point)
            else:
                var_below.append(point)
        
        if bool(var_above):
            var1_points.append(list(np.array(var_above).mean(axis=0).round().astype(int)))

        if bool(var_below):
            var2_points.append(list(np.array(var_below).mean(axis=0).round().astype(int)))

    

    ##########################
    ## Plotting var_points ##
    ##########################

    points_var1 = np.vstack((np.array(var1_points), list(orig_center)))
    points_var2 = np.vstack((np.array(var2_points), list(orig_center)))


    ## Plotting as different colors
    for point in points_var1:
        cv2.circle(contour_image_original,
                point,
                7,
                red_color,
                -1)

    yellow_color = (0,255,255)
    for point in points_var2:
        cv2.circle(contour_image_original,
                point,
                7,
                red_color,
                -1)  

    ##########################
    ## Apply polyfit to var ##
    ##########################

    poly_coef_var1 = np.polyfit(points_var1[:,0],
                                points_var1[:,1],
                                deg=poly_deg)
    poly_coef_var2 = np.polyfit(points_var2[:,0],
                                points_var2[:,1],
                                deg=poly_deg)

    x = np.linspace(np.min(points_var1[:,0])-x_less,np.max(points_var1[:,0])+x_plus,100)
    y = poly_coef_var1[0]*x**2 + poly_coef_var1[1]*x+poly_coef_var1[2]

    curve_points = np.column_stack((x,y)).astype(np.int32)

    cv2.polylines(curve_img, [curve_points], isClosed=False,color=blue_color,thickness=5)
    if fit_poly==True:
        contour_image_original = cv2.addWeighted(contour_image_original,1,curve_img,1,0)

    x = np.linspace(np.min(points_var2[:,0])-x_less,np.max(points_var2[:,0])+x_plus,100)
    y = poly_coef_var2[0]*x**2 + poly_coef_var2[1]*x+poly_coef_var2[2]

    curve_points = np.column_stack((x,y)).astype(np.int32)

    cv2.polylines(curve_img, [curve_points], isClosed=False,color=blue_color,thickness=5)
    if fit_poly==True:
        contour_image_original = cv2.addWeighted(contour_image_original,1,curve_img,1,0)


    return contour_image_original, poly_coef_mean, poly_coef_var1, poly_coef_var2


# #################################
# ## Apply smoothing to Contours ##
# #################################

# # Create an empty list to store smoothed contours
# smoothed_contours = []

# # Apply contour smoothing (contour approximation) to each selected contour
# epsilon = 50  # Adjust epsilon as needed
# for contour in selected_contours:
#     smoothed_contours.append(cv2.approxPolyDP(contour, epsilon, True))

# # print("Smoothed contours size:", smoothed_contours[0].shape)
# # Draw smoothed contours on the original image (or a copy of it)
# contour_image_smoothed = image.copy()
# cv2.drawContours(contour_image_smoothed, smoothed_contours, -1, (0, 255, 0), 2)
# cv2.circle(contour_image_smoothed, orig_center, 5, (255, 0, 0), -1)  # Draw red circle based on input


#########################
## Ask user for center ##
#########################

image_to_click = ImagePointPicker(img_path)
image_to_click.ask_user()
orig_center = image_to_click.clicked_point 

print("orig center:", type(orig_center), orig_center)

eps_smooth = 10
print("testing unsmoothed")
contour_image_original = learn_center_var(img_path=img_path,
                                          orig_center=orig_center)
print("success\n")

print("testing smoothed")
contour_image_smoothed = learn_center_var(img_path=img_path,
                                          orig_center=orig_center,
                                          smoothing=True,
                                          eps_smooth=eps_smooth)
print("success")

# Display the images side by side
comparison_image = cv2.hconcat([contour_image_original, contour_image_smoothed])

# Display the image with contours
cv2.imshow(f"Orig vs eps={eps_smooth} smoothed Contours", comparison_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

