import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def concentric_circle(orig_center,
                        img,
                        contour_smoothing = False,
                        contour_smoothing_eps = 50,
                        radii=50,
                        num_of_circs = 22,
                        fit_poly = True,
                        boundary_ring = False,
                        interior_ring = False,
                        interior_scale = 3/5,
                        rtol=1e-2,
                        atol=1e-6,
                        poly_deg = 2,
                        x_less = 600,
                        x_plus = 0,
                        mean_smoothing = True,
                        mean_smoothing_sigma =2):
    """
    To be applied to a single image. 

    Args:
        img (np.ndarray): numpy array of read image
    """
    # Check that original center has been declared
    if not isinstance(orig_center, tuple):
        raise TypeError(f"orig_center must be declared as {tuple}.\nPlease declare center or use find_center function.")

    # print(img.shape)
    # convert image to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ########################
    ## Apply Thresholding ##
    ########################

    # OTSU thresholding (automatically choose params)
    _, threshold = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ###################
    ## Find Contours ##
    ###################
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select n largest contours
    n=1
    contours = sorted(contours, key=cv2.contourArea,reverse=True)
    selected_contours = contours[:n]

    #############################
    ## Apply Contour Smoothing ##
    #############################

    # POTENTIALLY REMOVE SINCE IT REMOVES VERTICIES 
    if contour_smoothing == True:
        smoothed_contours = []

        for contour in selected_contours:
            smoothed_contours.append(cv2.approxPolyDP(contour, contour_smoothing_eps, True))
        
        selected_contours = smoothed_contours  

    ##############################
    ## Create Distance Array(s) ##
    ##############################
    # array of distance of each point on contour to original center
    contour_dist_list = []
    for contour in selected_contours:
        a = contour.reshape(-1,2)
        b = np.array(orig_center)
        contour_dist = np.sqrt(np.sum((a-b)**2,axis=1)) 
        contour_dist_list.append(contour_dist)

    ############################
    ## Draw contours on image ##
    ############################
    green_color = (0,255,0)
    red_color = (0,0,255)
    blue_color = (255,0,0)

    contour_img = img.copy()
    cv2.drawContours(contour_img, selected_contours,-1,green_color,2)
    cv2.circle(contour_img, orig_center,8,red_color,-1)

    ##############################
    ## Apply Concentric Circles ##
    ##############################

    # Initialize numpy array to store center
    points_mean = np.zeros(shape=(num_of_circs+1,2))
    points_mean[0] = orig_center

    # Plot first point on path 
    _, center = find_max_on_boundary(img_gray,
                                            orig_center,
                                            r=radii,
                                            rtol=rtol,
                                            atol=atol)
    # print("center_1:", center)
    points_mean[1] = center

    # draw rings if True
    if boundary_ring == True:
        # print("boundary_ring:", boundary_ring)
        cv2.circle(contour_img, orig_center, radii, red_color,1, lineType = cv2.LINE_AA)
    
    for step in range(2, num_of_circs+1):
        radius = radii*step

        # Draw interior_ring == True:
        if interior_ring == True:
            cv2.circle(contour_img,
                        center = center,
                        radius = int(radius*interior_scale),
                        color=red_color,
                        thickness=1,
                        lineType=cv2.LINE_AA)
        
        # Get center of next point
        error_occured = False
        try:
            # print(orig_center,center,radius,interior_scale,rtol,atol)
            _, center = find_next_center(array=img_gray,
                                                orig_center=orig_center,
                                                neig_center=center,
                                                r=radius,
                                                scale=interior_scale,
                                                rtol=rtol,
                                                atol=atol)
            # print(f"center_{step}:", center)
        except Exception as e:
            # print("empty search")
            error_occured = True
        
        if error_occured == True:
            break

        points_mean[step] = center

        # Draw boundary ring
        if boundary_ring == True:
            cv2.circle(contour_img,
                        center=orig_center,
                        radius=radius,
                        color=blue_color,
                        thickness=1,
                        lineType = cv2.LINE_AA)
    
    ############################
    ## Apply poly fit to mean ##
    ############################
    
    # Apply gaussian filtering to poitns in y direction
    if mean_smoothing == True:
        smooth_x = points_mean[:,0]
        smooth_y = gaussian_filter(points_mean[:,1], sigma=mean_smoothing_sigma)
        points_mean = np.column_stack((smooth_x,smooth_y))
    
    # Draw points on image
    new_points_mean = []
    for center in points_mean:
        for contour in selected_contours:
            if cv2.pointPolygonTest(contour, center,False)==1:
                new_points_mean.append(center)
                # Add additional check to see if point lies inside of contour.
                center = center.round().astype(int)
                cv2.circle(contour_img,center,7,red_color,-1)
    points_mean = np.array(new_points_mean).reshape(-1,2)
    poly_coef_mean = np.polyfit(points_mean[:,0], points_mean[:,1],deg=poly_deg)

    f_mean = lambda x: poly_coef_mean[0]*x**2 + poly_coef_mean[1]*x + poly_coef_mean[2]

    x = np.linspace(np.min(points_mean[:,0])-x_less,np.max(points_mean[:,0])+x_plus,100)
    y = f_mean(x)

    curve_img = np.zeros_like(contour_img)
    curve_points = np.column_stack((x,y)).astype(np.int32)

    cv2.polylines(curve_img,[curve_points], isClosed=False,color=red_color, thickness=5)
    if fit_poly==True:
        contour_img = cv2.addWeighted(contour_img,1,curve_img,1,0)


    ##############################
    ## Checking Variance points ##
    ##############################

    ## NEED TO COMPARE AGAINST SCRIPT GOING FORAWRD

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

            dist_mask = np.isclose(contour_dist, radius, rtol=1e-2)
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

    
    # Concatenate original center to both lists
    if bool(var1_points):
        points_var1 = np.vstack((np.array(var1_points), list(orig_center)))
    else:
        points_var1 = np.array([orig_center])
    
    if bool(var2_points):
        points_var2 = np.vstack((np.array(var2_points), list(orig_center)))
    else:
        points_var2 = np.array([orig_center])

    
    # Plotting poitns
    for point in points_var1:
        cv2.circle(contour_img,
                    point,
                    7,
                    blue_color,
                    -1)
    
    for point in points_var2:
        cv2.circle(contour_img,
                    point,
                    7,
                    blue_color,
                    -1)
        
    ##########################
    ## Apply polyfit to var ##
    ##########################

    # Add additional check for potentially underdetermined systems?
    # Possibly modify function
    poly_coef_var1 = np.polyfit(points_var1[:,0],
                                points_var1[:,1],
                                deg=poly_deg)
    
    poly_coef_var2 = np.polyfit(points_var2[:,0],
                                points_var2[:,1],
                                deg=poly_deg)
    
    # Plotting var 1
    x = np.linspace(np.min(points_var1[:,0])-x_less,np.max(points_var1[:,0])+x_plus,100)
    y = poly_coef_var1[0]*x**2 + poly_coef_var1[1]*x+poly_coef_var1[2]

    curve_points = np.column_stack((x,y)).astype(np.int32)
    
    cv2.polylines(curve_img, [curve_points], isClosed=False, color=blue_color, thickness=5)
    if fit_poly == True:
        contour_img = cv2.addWeighted(contour_img,1,curve_img,1,0)
    
    # Plotting var 2
    x = np.linspace(np.min(points_var2[:,0])-x_less,np.max(points_var2[:,0])+x_plus,100)
    y = poly_coef_var2[0]*x**2 + poly_coef_var2[1]*x + poly_coef_var2[2]

    curve_points = np.column_stack((x,y)).astype(np.int32)

    cv2.polylines(curve_img, [curve_points], isClosed=False,color=blue_color,thickness=5)
    if fit_poly == True:
        contour_img = cv2.addWeighted(contour_img,1,curve_img,1,0)
    
    return contour_img, poly_coef_mean, poly_coef_var1, poly_coef_var2

def find_next_center( array, orig_center, neig_center,r,scale=3/5,rtol=1e-3,atol=1e-6):
    # print("entered find_next_center")
    col, row = orig_center
    n, d = array.shape

    # generate grid of indicies from array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Get array of distances
    distances = np.sqrt((xx-col)**2 + (yy-row)**2)

    # Create mask for points on boundary (distances == r)
    # print(rtol, atol)
    boundary_mask = np.isclose(distances,r,rtol=rtol,atol=atol)

    # Appply to neighboring point
    col, row = neig_center

    # get array of new distances from pervious circle
    distances = np.sqrt((xx-col)**2 + (yy-row)**2)

    interior_mask = distances <= r*scale

    # print(np.argwhere(interior_mask==True))

    search_mask = boundary_mask & interior_mask

    # print(np.argwhere(search_mask == True))

    search_subset = array[search_mask]
    # print(search_subset)

    # print("made it here")

    max_value = np.max(search_subset)
    # print("made it here??")
    # print("max_value:", max_value)

    # find indicies of max element
    max_indices = np.argwhere(np.isclose(array,max_value) & search_mask)

    row, col = max_indices[0]

    max_indices = (col, row)

    return max_value, max_indices
    
def find_max_on_boundary( array, center,r,rtol=1e-3,atol=1e-6):
    col, row = center
    n, d = array.shape

    # Generate a grid of indicies for the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n)) 

    # Get Euclidean distance from each point on grid to center
    distances = np.sqrt((xx-col)**2+(yy-row)**2)

    # Create mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances,r,rtol=rtol,atol=atol)

    # Apply the boundary mask to the array to get the subset
    boundary_subset = array[boundary_mask]

    # Find the maximum value within the subset
    max_value = np.max(boundary_subset)

    # Find the indices of the maximum elements within the boundary 
    max_indices = np.argwhere(np.isclose(array, max_value) & boundary_mask)

    row, col = max_indices[0]
    
    max_indices = (col, row)

    return max_value, max_indices

def create_background_img(video_path, img_count):
    """
    Create background image for fixed subtraction method. 
    Args:
        img_count (int): number of initial frames to create average image to subtract from frames.
    Returns:
        background_img_np (np.ndarray): Numpy array of average image (in grayscale).
    """
    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    background_img_np = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float)

    k=0
    try:
        while ret:
            if k< img_count:
                background_img_np += cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            k+=1
            ret,frame = video_capture.read()
    except KeyboardInterrupt:
        pass
    finally:
        video_capture.release()
        # pass

    background_img_np = (background_img_np/img_count).astype(np.uint8)

    return background_img_np

def main():
    frame=4*119+34*3
    # frame =8*119
    print("frame:", frame)
    video_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_2/high_2.MP4"
    plume_leak_source = (1586, 1017)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame = cap.read()

    if ret:
       cv2.imshow("frame", frame)
       cv2.waitKey()
       cap.release()
       cv2.destroyAllWindows() 
    
    background_frame = create_background_img(video_path=video_path, img_count=4*119+34)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.subtract(frame,background_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    output = concentric_circle(orig_center=plume_leak_source, img=frame)
    frame = output[0]

    cv2.imshow("concentric circ", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




