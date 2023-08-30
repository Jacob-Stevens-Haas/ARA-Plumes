import os
import cv2
from moviepy.editor import VideoFileClip
import imageio
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm



# General Purpose functions
def count_files(directory: str, extension: str) -> int:
    """
    Return the number of items in directory ending with a certain extension.

    Args:
        directory (str): Directory path containing files
        extension (str): Extension of interest, e.g. "png", "jpg".
    
    Returns:
        int: Count for number of files in directory.
    
    """
    count = 0

    # Iterate over the files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has extension
        if filename.endswith('.'+extension):
            count += 1
    return count

def create_directory(directory):
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"Directory '{directory}' created.")
    # else:
    #     print(f"Directory '{directory}' already exists.")


# For extracting frames from video
def extract_all_frames(video_path, save_path="frames", extension="png"):
    """
    To extract all frames from a video.

    Args:
        video_path (str): path to video 
        save_path (str): folder name of save frames
        extension (str): file type to save frames as e.g., "png", "jpg"
    """
    clip = VideoFileClip(video_path)
    total_frames = int(clip.fps * clip.duration)

    frame_mag = len(str(total_frames))

    create_directory(save_path)

    for frame_number in range(total_frames):
        frame_time = frame_number / clip.fps
        frame = clip.get_frame(frame_time)

        # Modified Code to have consistent nomenclature
        if len(str(frame_number)) < frame_mag:
            num_of_lead_zeros = frame_mag - len(str(frame_number))
            frame_str = ''
            for i in range(num_of_lead_zeros):
                frame_str += '0'
            frame_number = frame_str+str(frame_number)
            
        if not isinstance(frame_number, str):
            frame_number = str(frame_number)

        frame_path = f"{save_path}/frame_"+frame_number+"."+extension
        imageio.imwrite(frame_path, frame)

# Functions for subtracting frames 
def create_id(id, magnitude):
    num_of_leading_zeros = magnitude - len(str(id))
    frame_str = ''
    for i in range(num_of_leading_zeros):
        frame_str +='0'
    frame_number = frame_str+str(id)
    return frame_number



def get_frame_ids(directory: str, extension: str = "png") -> list:
    """
    Return list of items in directory ending with a certain extension.

    Args:
        directory (str): Directory path containing files
        extension (str): Extension of interest, e.g. "png", "jpg".
    
    Returns:
        list: List of files in directory.
    
    """
    file_ids = []

    # Iterate over the files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has desired extension
        if filename.endswith('.'+extension):
            file_ids.append(filename)
            # print(file_ids)
    file_ids.sort()
    return file_ids

def fixed_average_subtract(frames_path,
                           background_range,
                           frames_range = None,
                           extension = "png",
                           save_path="fixed_avg_frames",
                           invert = False, 
                           normalize = False, 
                           grayscale = True):
    
    """
    Apply fixed average subtraction method to video frames. 

    Args:
        frames_path (str): directory that contains frames to apply subtraction. 
        background_range (list): 2 element list of start/end frame int for desired range to use as average background.
        frames_range (int, list, None): Range of frames to apply subtraction too. Default None use remaining frames after background_range.
        extension (str): extension type to search for in frames_path folder.
        save_path (str): folder name created to save subtracted frames too.
        invert (bool): To invert frame colors. 
        normalize (bool): To normalize grayscale values (i.e., 0 to 1 vals).
        grayscale (bool): Convert frames to grayscale type in open cv package

    Returns:
        saved frames in save_path folder.
                    
    I think Normalize might be busted--Will fix later.
    """
    
    # Get list of frames [frame_0000.png, ...]
    frames_id = get_frame_ids(directory=frames_path, extension=extension)
    
    # Create save_path if it does not exist
    create_directory(save_path)

    ###########################
    ## create background img ##
    ###########################
    
    # Get desired frame range of background from background_range
    if isinstance(background_range, list) and len(background_range)==2:
        start_frame, end_frame = background_range
    else:
        raise ValueError("background_range must be a 2 int list.")

    img_paths = [os.path.join(frames_path,frames_id[i]) for i in range(start_frame,end_frame+1)]
    img_count = len(img_paths)

    background_img_np = cv2.imread(img_paths[0]).astype(float)
    img_paths = img_paths[1:]
    for path in img_paths:
        background_img_np += cv2.imread(path).astype(float)
         
    background_img_np = (background_img_np/img_count).astype(np.uint8)
    if grayscale == True:
        background_img_np = cv2.cvtColor(background_img_np, cv2.COLOR_BGR2GRAY)

    ##########################################
    ## Apply subtraction to selected frames ##
    ##########################################

    # Get desired frame range from frames_range
    if frames_range is None:
        tot_frames = len(frames_id)
        start_frame = background_range[1]+1
        end_frame = tot_frames-1
    elif isinstance(frames_range, list) and len(frames_range)==2:
        start_frame, end_frame = frames_range
    elif isinstance(frames_range, int):
        tot_frames = len(frames_id)
        start_frame = frames_range
        end_frame = tot_frames-1
    else:
        raise ValueError("frames_range must be a 2 int list, single int, or None type  (uses remaining frames).")

    # Get magnitude of number of pictures for naming convention
    subtracted_mag = len(str(end_frame-start_frame))

    subtract_id = 0 
    for current_id in range(start_frame,end_frame+1):
        current_str = frames_id[current_id]
        current_path = os.path.join(frames_path,current_str)
        current_img_np = cv2.imread(current_path).astype(np.uint8)

        if grayscale == True:
            current_img_np = cv2.cvtColor(current_img_np, cv2.COLOR_BGR2GRAY)

        subtracted_img = cv2.subtract(current_img_np, background_img_np)

        # Invert image color (Just for visulazing)
        if invert == True:
            subtracted_img=np.invert(subtracted_img)

        # Normailze image 
        if normalize == True:
            # convert to float
            float_img = subtracted_img.astype(float)
            # subtracted_img = cv2.normalize(float_img, None, 0,1,cv2.NORM_MINMAX)
            image_normalized = cv2.normalize(float_img, None, 0, 1, cv2.NORM_MINMAX)
            image_mean = np.mean(image_normalized)
            image_std = np.std(image_normalized)
            subtracted_img = (image_normalized - image_mean) / image_std


        # Save subtracted image
        new_id = create_id(subtract_id,subtracted_mag)
        file_name = "subtract_"+new_id+"."+extension

        cv2.imwrite(os.path.join(save_path,file_name), subtracted_img)

        subtract_id +=1

def moving_average_subtract(frames_path: str,
                            frames_range,
                            median_length = 200,
                            extension = "png",
                            save_path="moving_avg_frames",
                            invert=False, 
                            grayscale = True):
    """
    Uses moving average subtract to isolate video and save frames to directory

    Args:
        frames_path (str): Directory path to frames to applying moving average
        frames_range (list): 2 element array containing start and end frame to apply subtracting 
        median_length (int): Number of frames to consider for subtraction 
        extension (str): Specifying format to save frames as, e.g., "png", "jpg"
        save_path (str): Directory to save subtracted frames
        invert (bool): True to invert color of images.

    """
    # Get list of frames [frame_0000.png, ...]
    frames_id = get_frame_ids(directory=frames_path, extension=extension)

    # Create directory to save subtracted frames 
    create_directory(save_path)

    subtracted_mag = len(str(frames_range[1]-frames_range[0]))

    subtract_id = 0
    for current_id in range(frames_range[0], frames_range[1]+1):
        background_id = current_id - median_length//2 # Probably need to change this so it's true median 

        img1_str = frames_id[background_id]
        img2_str = frames_id[current_id]

        img1_path = os.path.join(frames_path, img1_str)
        img2_path = os.path.join(frames_path, img2_str)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if grayscale == True:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img_subtracted = cv2.subtract(img1,img2)
        if invert==True:
            img_subtracted=np.invert(img_subtracted)

        # save subtracted to new folder 
        new_id = create_id(subtract_id,subtracted_mag)
        file_name = "subtract_"+new_id+"."+extension
        cv2.imwrite(os.path.join(save_path,file_name), img_subtracted)

        subtract_id+=1

##############################
## Edge detection functions ##
##############################

# ADD variables for hyperparameter tuning 
def edge_detect(frames_path,
                extension="png",
                save_path = "edge_frames",
                d = 10,
                sigmaColor = 20,
                sigmaSpace = 20,
                t_lower = 5,
                t_upper = 10):
    """
    Uses Bilaterial filtering in conjuction with opencv edge detection.
    """
    
    # Get list of frames [frame_0000.png, ...]
    frames_id = get_frame_ids(directory=frames_path, extension=extension)

    # Create directory to save subtracted frames 
    create_directory(save_path)

    frames_mag = len(str(len(frames_id)))

    count = 0
    for frame in frames_id:
        img_path = os.path.join(frames_path, frame)
        img = cv2.imread(img_path)

        # convert to gray 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Bilateral filter smoothing without removing edges.
        gray_filtered = cv2.bilateralFilter(gray,
                                            d=d,
                                            sigmaColor=sigmaColor,
                                            sigmaSpace=sigmaSpace)

        # Use Canny to get edge contours
        edges_filtered = cv2.Canny(gray_filtered,
                                   threshold1=t_lower,
                                   threshold2=t_upper)

        # save edge detected frame
        new_id = create_id(count, frames_mag)
        file_name = "edge_"+new_id+"."+extension
        cv2.imwrite(os.path.join(save_path,file_name), edges_filtered)

        count += 1
    return

########################
## Finding Plume Path ##
########################

# Class for having users pick initial center for plume detection

class ImagePointPicker:
    def __init__(self, img_path):
        self.img_path = img_path
        self.clicked_point = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            # print(f"Clicked at ({x}, {y})")
            cv2.destroyAllWindows()

    def ask_user(self):
        image = cv2.imread(self.img_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)

        cv2.imshow("Image", image_gray)

        while self.clicked_point is None:
            cv2.waitKey(1)

        # print("Clicked point:", self.clicked_point)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# Getting paths on each image
def find_max_on_boundary(array, center,r,rtol=1e-3,atol=1e-6):

    col, row = center
    n, d = array.shape

    # Generate a grid of indices for the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n)) 

    # Calculate the Euclidean distance from each point to the center
    # Create array of distances from center 
    distances = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)

    # Create a mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances, r, rtol=rtol,atol=atol)
    # boundary_mask = np.isclose(distances,r)

    # Apply the boundary mask to the array to get the subset
    boundary_subset = array[boundary_mask]

    # Find the maximum value within the subset
    max_value = np.max(boundary_subset)

    # Find the indices of the maximum elements within the boundary
    max_indices = np.argwhere(np.isclose(array, max_value) & boundary_mask)

    row, col = max_indices[0]

    max_indices = (col, row)


    # Return the maximum value and its locations
    return max_value, max_indices

def find_next_center(array, orig_center, neig_center, r,scale=3/5, rtol=1e-3,atol=1e-6):
    col, row = orig_center
    n, d = array.shape

    # print("curr center:", (col,row))

    # generate grid of indicies from the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Get array of distances 
    distances = np.sqrt((xx-col)**2 + (yy-row)**2)

    # Create a mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances,r,rtol=rtol,atol=atol)
    # boundary_mask = np.isclose(distances,r)


    # create interion on previous circle
    col, row = neig_center

    # print("prev center:", (col,row))

    # get array of new distnaces from previous circle
    distances = np.sqrt((xx-col)**2 + (yy-row)**2)

    # Create interior mask
    interior_mask = distances <= r*scale

    search_mask = boundary_mask & interior_mask

    search_subset = array[search_mask]

    # print("search subset:",search_subset)

    max_value = np.max(search_subset)

    # find indicies of max element
    max_indicies = np.argwhere(np.isclose(array,max_value) & search_mask)

    row, col = max_indicies[0]

    max_indicies = (col,row)

    return max_value, max_indicies

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
    """Applyes concentric cricles to single image learning both mean and edges of plumes."""
    
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

    # check if list is empty 
    if bool(var1_points):
        points_var1 = np.vstack((np.array(var1_points), list(orig_center)))
    else:
        points_var1 = np.array(orig_center)

    if bool(var2_points):
        points_var2 = np.vstack((np.array(var2_points), list(orig_center)))
    else:
        points_var2 = np.array(orig_center)
    # points_var1 = np.vstack((np.array(var1_points), list(orig_center)))
    # points_var2 = np.vstack((np.array(var2_points), list(orig_center)))


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

    # Add additional check for potentially underdetermined systems
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
 

def find_center_of_mass(frames_path: str,
                        center: tuple,
                        frames_range = None,
                        save_path = "learned_path_frames",
                        radii=8,
                        num_of_circs = 17,
                        extension='png',
                        boundary_ring = True,
                        interior_ring = True,
                        rtol = 1e-2,
                        atol = 1e-6,
                        scale=3/5,
                        fit_poly=True,
                        x_plus =0,
                        x_less = 0,
                        smoothing=False,
                        eps_smooth=10):
    """
    Generate frames of learned path on background subtract frames.

    Args:
        frames_path (str): directory path to background subtract frames to learn path on.
        center (tuple): 2 element tuple of frame coordinates of plume leak origin (col, row)
        frames_range (int, list, None): Frames range to apply learning to from frames_path. Default None uses entire range.
        save_path (str): directory used/created to stored new frames.
        radii (int): Search radii step for concentric circles.
        num_of_circs (int): Number of concentric rings used for search. 
        extension (str): Extension to search for in frames_path. Extension used to save new frames.
        boundary_ring (bool): Toggle plotting of boundary rings used for search.
        interior_ring (bool): Toggle plotting of interion rings used for search.
        rtol (float): relative numerical tolerance for radii calculations. Used in np.isclose.
        atol (float): Absolute numerical tol for radii calculations. Used in np.isclose.
        scale (float): Scaled size of boundary ring used for interior ring radii. 
        fit_poly (bool): Toggle plotting of learned curves.
    Returns:
        Frames of learned paths. 
        poly_coef (np.array): array containing learned poly coefficients for each timestep.

    """
    # Get list of frames [frame_0000.png, ...]
    frames_id = get_frame_ids(directory=frames_path, extension=extension)

    # Create directory to save subtracted frames 
    create_directory(save_path)

    # For naming convection
    frames_mag = len(str(len(frames_id))) # might need to move this later - actually think this is fine

    # Select appropriate frame range - default None is all frames
    if frames_range is None:
        count = 0
    elif isinstance(frames_range, list) and len(frames_range)==2:
        start_frame, end_frame = frames_range
        frames_id = frames_id[start_frame:end_frame]
        count = start_frame
    elif isinstance(frames_range, int):
        start_frame = frames_range
        frames_id = frames_id[start_frame:]
        count = start_frame
    else:
        raise ValueError("frames_range must be a 2 int list, or a single int.")
    
    # Instantiate numpy array to store poly coeff
    poly_deg = 2
    poly_coef_mean_array = np.zeros(shape=(len(frames_id),poly_deg+1))
    poly_coef_var1_array = np.zeros(shape=(len(frames_id),poly_deg+1))
    poly_coef_var2_array = np.zeros(shape=(len(frames_id),poly_deg+1))


    # Retain original center coordinates
    orig_center = center # MIGHT NOT NEED TO DO THIS ANYMORE
    i = 0
    for frame in frames_id:
        img_path = os.path.join(frames_path,frame)

        # apply mean and var learning
        img, poly_coef_mean, poly_coef_var1, poly_coef_var2 = learn_center_var(img_path=img_path,
                                                                                orig_center=orig_center,
                                                                                smoothing=smoothing,
                                                                                eps_smooth=eps_smooth,
                                                                                radii=radii,
                                                                                num_of_circs=num_of_circs,
                                                                                fit_poly=fit_poly,
                                                                                boundary_ring=boundary_ring,
                                                                                interior_ring=interior_ring,
                                                                                scale=scale,
                                                                                rtol=rtol,
                                                                                atol=atol,
                                                                                poly_deg=poly_deg,
                                                                                x_less=x_less,
                                                                                x_plus=x_plus)
        poly_coef_mean_array[i] = poly_coef_mean
        poly_coef_var1_array[i] = poly_coef_var1
        poly_coef_var1_array[i] = poly_coef_var2

        # save path frame
        new_id = create_id(count, frames_mag)
        file_name = "path_"+new_id+"."+extension
        cv2.imwrite(os.path.join(save_path,file_name),img)

        count += 1    
        i +=1
        print(f"{file_name} success.")

        # img = cv2.imread(img_path)

        # # convert to gray
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # Instantiate numpy array to store centers
        # points = np.zeros(shape=(num_of_circs+1,2))
        # points[0] = orig_center

        # # pick and draw initial center (in blue)
        # # center = center
        # cir_radius = 3
        # blue_color = (255,0,0)
        # thickness = -1

        # # Plot original center
        # cv2.circle(img,orig_center, cir_radius, blue_color,thickness)

        # # Select and plot first point on path
        # _, center = find_max_on_boundary(array=gray, center=orig_center, r=radii, rtol=rtol,atol=atol)
        # points[1] = center

        # # plot red circle
        # cir_radius = 3
        # red_color = (0,0,255)
        # thickness = -1
        # cv2.circle(img,center,cir_radius,red_color,thickness)

        # # Draw the ring
        # if boundary_ring == True:
        #     cv2.circle(img,orig_center, radii, red_color, thickness=1,lineType=cv2.LINE_AA)

        # for step in range(2, num_of_circs+1):
        #     radius = radii*step

        #     # Draw interior ring
        #     if interior_ring == True:
        #         cv2.circle(img,
        #                    center = center,
        #                    radius = int(radius*scale),
        #                    color = blue_color,
        #                    thickness = 1,
        #                    lineType = cv2.LINE_AA)
                
        #     # Get center of next point
        #     ## Throw in try catch error --> break out of loop if does not work.
        #     error_occured = False
        #     try:
        #         _, center = find_next_center(array=gray,
        #                                     orig_center=orig_center,
        #                                     neig_center=center,
        #                                     r=radius,
        #                                     scale=scale,
        #                                     rtol=rtol,
        #                                     atol=atol)
        #     except Exception as e:
        #         print("empty search")
        #         error_occured = True
            
        #     if error_occured == True:
        #         break
        
        #     points[step]=center
            
        #     #draw next red point
        #     cv2.circle(img,center,cir_radius,red_color,thickness)

        #     # draw boundary ring
        #     if boundary_ring == True:
        #         cv2.circle(img,center=orig_center, radius=radius,color=red_color,thickness=1,lineType=cv2.LINE_AA)

        # # Learn polynomial coeff
        # poly_coef = np.polyfit(points[:,0],points[:,1], deg=poly_deg)
        # poly_coef_mean_array[i] = poly_coef
        
        # # x_plus = 50
        # # x_plus = 0
        # x = np.linspace(np.min(points[:,0])-x_less,np.max(points[:,0])+x_plus,100)
        # y = poly_coef[0]*x**2 + poly_coef[1]*x+poly_coef[2]

        # curve_img = np.zeros_like(img)
        # curve_points = np.column_stack((x,y)).astype(np.int32)

        # cv2.polylines(curve_img, [curve_points], isClosed=False,color = (255,0,0), thickness=5)

        # if fit_poly == True:
        #     img = cv2.addWeighted(img,1,curve_img,1,0) 
        
        # # save path frame
        # new_id = create_id(count, frames_mag)
        # file_name = "path_"+new_id+"."+extension
        # cv2.imwrite(os.path.join(save_path,file_name),img)

        # count += 1    
        # i +=1
        # print(f"{file_name} success.")
    return poly_coef_mean_array, poly_coef_var1_array, poly_coef_var2_array

#####################
## Post Processing ##
#####################

# For saving frames as video
def create_video(directory, output_file, fps=15, extension="png", folder = "movies"):
    """
    Create video from selected frames 

    Args:
        directory (str): Directory of where to access png files
        output_file (str): path and name of file to save, e.g., [path/]video.mp4
        fps (int): Specify the frames per second for video. 
    """
    # Create directory to store movies
    create_directory(folder)

    # Get the list of PNG files in the directory
    png_files = sorted([file for file in os.listdir(directory) if file.endswith('.'+extension)])

    # Get the first image to retrieve its dimensions
    first_image = cv2.imread(os.path.join(directory, png_files[0]))
    height, width, _ = first_image.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each image to the video writer
    for png_file in png_files:
        image_path = os.path.join(directory, png_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the video writer and close the file
    video_writer.release()

def create_gif(frames_dir: str,
               duration: int,
               frames_range: list = None,
               rotate_ang: float = 0,
               gif_name: str = "animation",
               extension: str = "png"):
    """
    Create GIF from specified frames directory. 

    Args:
        frames_dir (str): directory path containing frames
        duration (int): Number of milliseconds 
        frames_range (list): frames of interest for gif
        rorate_ang (float): counter clock-wise
        gif_name (str): name to be given
        extension (str): extension type to search for in directory 
    """
    def add_forward_slash(string):
        if not string.endswith('/'):
            string += '/'
        return string
    
    # start_frame, end_frame = frames_range
    
    # Add '/' to path if it does not already exist
    frames_dir = add_forward_slash(frames_dir)

    # Get all files with specified extension in frames_dir
    frame_paths = sorted(glob.glob(frames_dir + "*."+extension))

    # Select appropriate frame range - default None is all frames
    if frames_range is None:
        frame_paths = frame_paths
    elif isinstance(frames_range, list) and len(frames_range)==2:
        start_frame, end_frame = frames_range
        frame_paths = frame_paths[start_frame:end_frame]
    elif isinstance(frames_range, int):
        start_frame = frames_range
        frame_paths = frame_paths[start_frame:]
    else:
        raise ValueError("frames_range must be a 2 int list, or a single int.")

    # Instantiate list to store image files
    frames = []

    # Iterate over each frame in file
    for path in tqdm(frame_paths):
        # path = frame_paths[i]

        # Open the frame as image
        image = Image.open(path)

        # Append the image to the frames list
        frames.append(image.rotate(rotate_ang))
    
    # Save frames as animated gif
    frames[0].save(gif_name+".gif",
                    format="GIF", 
                    append_images=frames[1:],
                    duration = duration,
                    save_all = True,
                    loop=0)
    