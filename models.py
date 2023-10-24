import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes')
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# For dislaying clips in Jupyter notebooks 
# from IPython.display import Image, display 
import IPython

import time
import cv2

class plume_results():
    pass

class PLUME():
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_width = int(self.video_capture.get(3))
        self.frame_height = int(self.video_capture.get(4))
        self.fps = int(self.video_capture.get(5))
        self.tot_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.mean_poly = None
        self.var1_poly = None
        self.var2_poly = None
        self.orig_center = None
        self.count = None
    
    def display_frame(self, frame: int):
        cap = self.video_capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()

        if ret:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
    
    def background_subtract(self,
                            img_range: list,
                            subtraction_method: str = "fixed",
                            fixed_range: int = None,
                            save: bool = False,
                            save_path: str = "subtraction",
                            extension: str = "mp4",
                            display_vid: bool = True):
        
        #######################
        ## Fixed Subtraction ##
        #######################

        # Create background image 
        if subtraction_method == "fixed" and isinstance(fixed_range,int):
            print("Creating background image...")
            background_img_np = self.create_background_img(img_count = fixed_range)
            # print("back shape:", background_img_np.shape)
            print("done.")
        else:
            raise TypeError("fixed_range must be a positive int for fixed subtraction method.")
        
        self.video_capture = cv2.VideoCapture(self.video_path)

        video = self.video_capture

        # grab vieo info for saving new file 
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        frame_rate = int(video.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Possibly resave video
        if save:
            clip_title = save_path+"."+extension
            out = cv2.VideoWriter(clip_title, fourcc, frame_rate, (frame_width, frame_height),1)
    
        if display_vid == True:
            display_handle = IPython.display.display(None, display_id = True)

        # Loop over frames and apply tracking to img_range
        init_frame, fin_frame = img_range

        for _ in range(init_frame):
            _ = video.read()

        for k in tqdm(range(init_frame,fin_frame)):
            ret, frame = video.read()

            if not ret:
                print("break reached")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.subtract(frame, background_img_np)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


            _, frame = cv2.imencode('.jpeg', frame)

            if display_vid == True:
                display_handle.update(IPython.display.Image(data=frame.tobytes()))
        

        video.release()
        if save:
            out.release()


        # if display_vid == True:
        #     display_handle.update(None)

        #############################
        ## Moving Average Subtract ##
        #############################
        if subtraction_method == "moving_avg":
            print()


        return

    def get_center(self, frame = None):
        """
        QUITE BROKEN RIGHT NOW.
        Allows users to easily get coordinate point values from image. 
        Args:
            frame (int): Which frame to select from. Default is None type which gives middle frame of video. 
        Returns:
            orig_center (tuple): Assigns tuple of selected coordinate to self.orig_center
        """
        video_capture = self.video_capture
        tot_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_id = tot_frames // 2
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_id) 

        ret, frame = video_capture.read()

        video = self.video_capture
        ret, frame = video.read()
        

        # cid = None

        # if ret:
        #     plt.figure()
        #     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        #     plt.title('Click image to select point')

        #     def on_image_click(event):
        #         x = int(event.xdata)
        #         y = int(event.ydata)
        #         print(f"Selected point: ({x}, {y})")
        #         plt.disconnected(cid)
        #         plt.close()
            
        #     cid = plt.gcf().canvas.mpl_connect('button_press_event', on_image_click)

        #     # Display the plot
        #     IPython.display.display(plt.gcf())
        #     IPython.display.clear_output(wait=True)
        
        # video_capture.release()
        

        if ret:
            plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        # allow users to select point 
        print("made it to pre selected points")
        selected_point = plt.ginput(n=1,timeout=0)
        print("made it to after selection.")
        if selected_point:
            x,y = selected_point[0]
            self.orig_center = (x,y)
            print(f"Selected point: ({x}, {y})")
        else:
            print("No point selected.")
        return

    def extract_frames(self, extension: str = "jpg"):
        video_path = self.video_path
    
    def train(self,
            img_range: list=None,
            subtraction_method: str = "fixed",
            fixed_range: int = None,
            gauss_space_blur = True,
            gauss_kernel_size = 81,
            gauss_space_sigma = 15,
            gauss_time_blur = True,
            gauss_time_window = 5,
            gauss_time_sigma = 1,
            extension = "mp4",
            save_path = None,
            display_vid = True,
            mean_smoothing = True,
            mean_smoothing_sigma =2):
        """
        Apply connetric circles to frames range and providing timeseries of learned polynomials.

        Args:
            img_range (list): Range of images to apply subtraction and concentric circles too.
            subtraction_method (str): Method used to apply subtraction.
            fixed_range (int): Range of images to use as background image for subtraction.
            gauss_space_blur (bool): Apply GaussianBlur in space
            gauss_kernel_size (odd int): size of kernel for GaussianBlur (must be odd)
            gauss_space_sigma (int?): standard variation of kernel size.
            extension (str): file format video is saved as.
            save_path (str): Path/name of video. Default None will not save video.
            display_vid (bool): Display concentric circles in jupyter notebook window
            mean_smoothing (bool): Additional gaussian smoothing for leaning concentric circle points
            mean_smoothing_sigma (int): standard variation for gaussian kernel smoother of mean_smoothing
        """
        # Create background image 
        if subtraction_method == "fixed" and isinstance(fixed_range,int):
            print("Creating background image...")
            background_img_np = self.create_background_img(img_count = fixed_range)
            # print("back shape:", background_img_np.shape)
            print("done.")
        else:
            raise TypeError("fixed_range must be a positive int for fixed subtraction method.")
        
        # Reread after releasing from create_bacground_img
        self.video_capture = cv2.VideoCapture(self.video_path)

        video = self.video_capture
        ret, frame = video.read()
        # print("first ret:", ret)

        # grab vieo info for saving new file 
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        frame_rate = int(video.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Possibly resave video
        if isinstance(save_path,str):
            clip_title = save_path+"."+extension
            color_true=1
            out = cv2.VideoWriter(clip_title, fourcc, frame_rate, (frame_width, frame_height),color_true)

        if display_vid == True:
            display_handle = IPython.display.display(None, display_id = True)
        
        # Select desired video range
        if isinstance(img_range,list) and len(img_range)==2:
            if img_range[1] > self.tot_frames:
                print("Img_range exceeds total number of frames...")
                print(f"Using max frame count {self.tot_frames}")
                init_frame = img_range[0]
                fin_frame = self.tot_frames -1
            else:
                init_frame, fin_frame = img_range
        elif isinstance(img_range, int):
            init_frame = img_range
            fin_frame = self.tot_frames - 1
        elif isinstance(img_range, None):
            init_frame = fixed_range
            fin_frame = self.tot_frames -1
        
        # Initialize poly arrays
        mean_array = np.zeros((fin_frame-init_frame,3))
        var1_array = np.zeros((fin_frame-init_frame,3))
        var2_array = np.zeros((fin_frame-init_frame,3))

        # Check for Gaussian Time Blur
        if gauss_time_blur == True:
            if not isinstance(gauss_time_window,int):
                raise Exception("window must be a positive odd int.")
            if not gauss_time_window%2==1:
                raise Exception("window must be a positive odd int.") 

            # Create gaussian_time_blur variables
            buffer = int(gauss_time_window/2)
            frames_to_average = list(np.zeros(gauss_time_window))
            gauss_time_weights = [np.exp(-(x-buffer)**2/(2*gauss_time_sigma**2)) for x in range(gauss_time_window)]
            gauss_time_weights /= np.sum(gauss_time_weights)
        else:
            buffer = 0   

        # Ignore first set of frames not in desired range
        for _ in tqdm(range(init_frame-buffer)):
            # print(i)
            ret, frame = video.read()
            _, frame = cv2.imencode('.jpeg', frame)
            if display_vid == True:
                display_handle.update(IPython.display.Image(data=frame.tobytes()))
        
        i=0
        for _ in tqdm(range(2*buffer)):
            ret, frame = video.read()
            # convert frame to gray 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply subtraction (still in gray)
            frame = cv2.subtract(frame, background_img_np)
            # Convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  
            frames_to_average[i]=frame
            i +=1            


        for k in tqdm(range(fin_frame-init_frame)):
            ret, frame = video.read()
            # print(i)

            # Check if video_capture was read in correctly
            if not ret:
                break
            
            # convert frame to gray 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply subtraction
            frame = cv2.subtract(frame, background_img_np)

            # Convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if gauss_time_blur == True:
                frames_to_average[i]=frame
                # Apply gaussian filter over windows
                frame = self.BGRframes_dot_weights(frames_to_average,
                                                   gauss_time_weights)
                # Move frames over left
                frames_to_average[:-1]=frames_to_average[1:]
                i=-1             

            if gauss_space_blur == True:
                kernel_size = (gauss_kernel_size,gauss_kernel_size)
                sigma = gauss_space_sigma

                frame = cv2.GaussianBlur(frame, kernel_size, sigma, sigma)

            out_data = self.concentric_circle(frame,
                                              mean_smoothing=mean_smoothing,
                                              mean_smoothing_sigma=mean_smoothing_sigma)

            frame = out_data[0]
            mean_array[k] = out_data[1]
            var1_array[k] = out_data[2]
            var2_array[k] = out_data[3]

            if isinstance(save_path, str):
                out.write(frame)
                # print("frames saved..")
            _, frame = cv2.imencode('.jpeg', frame)

            if display_vid == True:
                display_handle.update(IPython.display.Image(data=frame.tobytes()))
            
        # video.release()
        if isinstance(save_path, str):
            # print("we got eem")
            video.release()
            out.release()


        if display_vid == True:
            display_handle.update(None)
        
        self.mean_poly = mean_array
        self.var1_poly = var1_array
        self.var2_poly = var2_array

        return mean_array, var1_array, var2_array


    def concentric_circle(self,
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
        if not isinstance(self.orig_center, tuple):
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
            b = np.array(self.orig_center)
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
        cv2.circle(contour_img, self.orig_center,8,red_color,-1)

        ##############################
        ## Apply Concentric Circles ##
        ##############################

        # Initialize numpy array to store center
        points_mean = np.zeros(shape=(num_of_circs+1,2))
        points_mean[0] = self.orig_center

        # Plot first point on path 
        _, center = self.find_max_on_boundary(img_gray,
                                              self.orig_center,
                                              r=radii,
                                              rtol=rtol,
                                              atol=atol)
        # print("center_1:", center)
        points_mean[1] = center

        # draw rings if True
        if boundary_ring == True:
            # print("boundary_ring:", boundary_ring)
            cv2.circle(contour_img, self.orig_center, radii, red_color,1, lineType = cv2.LINE_AA)
        
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
                # print(self.orig_center,center,radius,interior_scale,rtol,atol)
                _, center = self.find_next_center(array=img_gray,
                                                  orig_center=self.orig_center,
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
                           center=self.orig_center,
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
        
        ###########################################
        ## Check if points fall within a contour ##
        ###########################################
    
        new_points_mean = []
        for center in points_mean:
            for contour in selected_contours:
                if cv2.pointPolygonTest(contour, center, False)==1:
                    new_points_mean.append(center)
                    center = center.round().astype(int)
                    cv2.circle(contour_img,center,7,red_color,-1)
        
        if bool(new_points_mean):
            points_mean = np.array(new_points_mean).reshape(-1,2)
        
        # We are going to learn two different polynomails that are paramertized in r -> (x(r),y(r))
        # Also make sure orig center is still in there? might move where we add it to this line
        # Create r line r_vals = [radii*i for i in range(len(points_mean))]

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
            points_var1 = np.vstack((np.array(var1_points), list(self.orig_center)))
        else:
            points_var1 = np.array([self.orig_center])
        
        if bool(var2_points):
            points_var2 = np.vstack((np.array(var2_points), list(self.orig_center)))
        else:
            points_var2 = np.array([self.orig_center])

        
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
    
    def find_next_center(self, array, orig_center, neig_center,r,scale=3/5,rtol=1e-3,atol=1e-6):
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
        
    def find_max_on_boundary(self, array, center,r,rtol=1e-3,atol=1e-6):
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
    
    def create_background_img(self, img_count):
        """
        Create background image for fixed subtraction method. 
        Args:
            img_count (int): number of initial frames to create average image to subtract from frames.
        Returns:
            background_img_np (np.ndarray): Numpy array of average image (in grayscale).
        """
        ret, frame = self.video_capture.read()
        # print("ret:", ret)
        background_img_np = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float)

        k=0
        try:
            while ret:
                if k< img_count:
                    background_img_np += cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
                k+=1
                ret,frame = self.video_capture.read()
        except KeyboardInterrupt:
            pass
        finally:
            self.video_capture.release()
            # pass

        background_img_np = (background_img_np/img_count).astype(np.uint8)

        return background_img_np

    def BGRframes_dot_weights(self,frames,weights):
        """
        Takes list of numpy arrays (imgs) and dots them with vector of weights 

        Args:
            frames (list): List of numpy ararys in BGR format
            weights (np.ndarray): vector of weights summing to one
        """
        a = np.array(frames)
        b = np.array(weights)[:,np.newaxis,np.newaxis,np.newaxis] # Might need to change if working with gray images
        c = np.round(np.sum(a*b,axis=0)).astype(np.uint8)
        return c 

    def clip_video(self, init_frame, fin_frame, extension: str = "mp4", display_vid: bool = True, save_path: str = "clipped_video"):
        video = self.video_capture
        print(type(video.read()))
        ret, frame = video.read()
        # return frame

        # grab vieo info for saving new file 
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        frame_rate = int(video.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Possibly resave video
        clip_title = save_path+"."+extension
        out = cv2.VideoWriter(clip_title, fourcc, frame_rate, (frame_width, frame_height),0)

        if display_vid == True:
            print("display_handle defined.")
            display_handle = IPython.display.display(None, display_id = True)
        
        # Loop results in the writing of B&W shortned video clip
        k=0
        try:
            while ret:
                if k<fin_frame and k>=init_frame:
                    # print("entered update")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out.write(frame)
                    _, frame = cv2.imencode('.jpeg', frame) # why is this jpeg?
                    if display_vid == True:
                        # print("update display")
                        display_handle.update(IPython.display.Image(data=frame.tobytes()))
                print("k:", k)
                k+=1
                ret, frame = video.read()
        except KeyboardInterrupt:
            pass
        finally:
            print("finished")
            # video.release()
            out.release()
            if display_vid == True:
                display_handle.update(None)

        return
    
    



def main():   
    video_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/high_1.MP4"


    clip = VideoFileClip(video_path)

    test_num =15

    t0 = time.time()
    fps = clip.fps
    total_frames = int(fps*clip.duration)
    frames = list(np.arange(test_num)*100+100)
    for frame in tqdm(frames):
        frame_time = frame/fps
        output = clip.get_frame(frame_time)
    t1 = time.time()
    print("time:", t1-t0,"\n")


    # We are going to use the cv2 method (faster)
    video_capture = cv2.VideoCapture(video_path)
    t0 = time.time()

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames //2
    frames = list(np.arange(test_num)*100+100)  
    for frame in tqdm(frames):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
        _,output = video_capture.read()


    t1 = time.time()
    print("time:", t1-t0)

    print("new test")

if __name__ == "__main__":
    main()