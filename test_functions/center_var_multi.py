#############################################
## Apply center_and_var to multiple frames ##
#############################################

# 160 - 260

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from tqdm import tqdm
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes/')

from utils import create_id, find_max_on_boundary, find_next_center, get_frame_ids, create_directory

##############################
## Get desired frames range ##
#############################
save_path = "Aug_10_frames"
frames_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames"
extension = "png"

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
x_plus = 0 #50
blue_color = (255,0,0)
red_color = (0,0,255)

# Get frames id list
frames_id = get_frame_ids(directory=frames_path,
                          extension=extension)

# Create save directory
create_directory(save_path)

# naming convetion
frames_mag = len(str(len(frames_id)))

# print(frames_mag)
start_frame = 528
end_frame = 628-1

orig_center = (1590,1000)
radii = 50
num_of_circs = 22

i=0
count = start_frame
for frame in tqdm(frames_id[start_frame:end_frame+1]):
    img_path = os.path.join(frames_path, frame)
    image = cv2.imread(img_path)

    # convert to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OTSU thresholding (automatically choose params)
    _, threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Simple Thresholding
    # threshold_value = 60
    # _, threshold = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find Contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select n largest contours
    n=1
    contours = sorted(contours,key=cv2.contourArea, reverse = True)
    selected_contours = contours[:n]

    # Creating distances on contours
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
    x_plus = 0 #50
    blue_color = (255,0,0)
    red_color = (0,0,255)

    # Instatiate numpy array to store mean points
    points_mean = np.zeros(shape=(num_of_circs+1,2))
    points_mean[0] = orig_center

    # Instantie list to store variance points
    var_points = []

    # Plot first point on path
    _, center = find_max_on_boundary(image_gray, orig_center,radii, rtol=rtol,atol=atol)
    points_mean[1]=center

    ##### Check variance intersection
    for i in range(len(selected_contours)):
        contour_dist = contour_dist_list[i]
        contour = selected_contours[i]

        dist_mask =np.isclose(contour_dist,radii, rtol=1e-2)
        intersection_points = contour.reshape(-1,2)[dist_mask]

        for point in intersection_points:
            var_points.append(list(point))

    # Draw rings
    if boundary_ring == True:
        cv2.circle(contour_image_original, orig_center, radii, (0,0,255),1, lineType = cv2.LINE_AA)

    for step in range(2, num_of_circs+1):
        radius = radii*step

        #################################
        ## Check variance intersection ##
        #################################
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
        
        points_mean[step] = center

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
            
    ############################
    ## Apply Poly fit to mean ##
    ############################
    poly_coef_mean = np.polyfit(points_mean[:,0], points_mean[:,1],deg=poly_deg)

    x = np.linspace(np.min(points_mean[:,0])-x_less,np.max(points_mean[:,0])+x_plus,100)
    y = poly_coef_mean[0]*x**2 + poly_coef_mean[1]*x+poly_coef_mean[2]

    curve_img = np.zeros_like(contour_image_original)
    curve_points = np.column_stack((x,y)).astype(np.int32)

    cv2.polylines(curve_img, [curve_points], isClosed=False,color=blue_color,thickness=5)
    if fit_poly==True:
        contour_image_original = cv2.addWeighted(contour_image_original,1,curve_img,1,0)


    ##########################
    ## Splitting var_points ##
    ##########################

    var_points_np = np.array(var_points)
    # print(type(var_points_np),var_points_np.shape)

    x_var = var_points_np[:,0]
    y_var = var_points_np[:,1]

    poly_out = poly_coef_mean[0]*x_var**2 + poly_coef_mean[1]*x_var + poly_coef_mean[2]

    above_mask = y_var >= poly_out
    below_mask = ~above_mask

    points_var1 = np.vstack((var_points_np[above_mask], list(orig_center)))
    points_var2 = np.vstack((var_points_np[below_mask], list(orig_center)))


    ## Plotting as different colors
    for point in points_var1:
        cv2.circle(contour_image_original,
                point,
                7,
                red_color,
                -1)

    for point in points_var2:
        cv2.circle(contour_image_original,
                point,
                7,
                blue_color,
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

    #################
    ## Save Frames ##
    #################

    new_id = create_id(count, frames_mag)
    file_name = "path_"+new_id+"."+extension
    cv2.imwrite(os.path.join(save_path,file_name),contour_image_original)

    count +=1
    i+=1
    # print(f"{file_name} success.")