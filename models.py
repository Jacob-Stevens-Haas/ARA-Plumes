import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes')
from utils import *
from tqdm import tqdm

import time
import cv2

class PLUME():
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.mean_poly = None
        self.var1_poly = None
        self.var2_poly = None
        self.orig_center = None
    

    def extract_frames(self, extension: str = "jpg"):
        video_path = self.video_path
    
    def concentric_circle(self,
                          img,
                          contour_smoothing = True,
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
        """
        # convert image to gray
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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
        red_color = (255,0,0)
        blue_color = (0,0,255)

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
        points_mean[1] = center

        # draw rings if True
        if boundary_ring == True:
            cv2.circle(contour_img, self.orig_center, radii, (0,0,255),1, lineType = cv2.LINE_AA)
        
        for step in range(2, num_of_circs+1):
            radius = radii*step

            # Draw interior_ring == True:
            if interior_ring == True:
                cv2.circle(contour_img,
                           center = center,
                           radius = int(radius*interior_scale),
                           color=blue_color,
                           thickness=1,
                           lineType=cv2.LINE_AA)
            
            # Get center of next point
            error_occured = False
            # try:
            #     _, center = (5,5)
        


        return
    
    def find_next_center(self, array, orig_center, neig_center,r,scale=3/5,rtol=1e-3,atol=1e-6):
        col, row = orig_center,
        n, d = array.shape

        # generate grid of indicies from array
        xx, yy = np.meshgrid(np.arange(d), np.arange(n))

        # Get array of distances
        distances = np.sqrt((xx-col)**2 + (yy-row)*2)

        # Create mask for points on boundary (distances == r)
        boundary_mask = np.isclose(distances,r,rtol=rtol,atol=atol)

        # Appply to neighboring point
        col, row = neig_center

        interior_mask = distances <= r*scale

        search_mask = boundary_mask & interior_mask

        search_subset = array[search_mask]

        max_value = np.max(search_subset)

        # find indicies of max element
        max_indicies = np.argwhere(np.isclose(array,max_value) & search_mask)

        row, col = max_indicies[0]

        max_indicies = (col, row)

        return max_value, max_indicies
        return
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
        max_indicies = np.argwhere(np.isclose(array,max_value) & boundary_mask)

        row, col = max_indicies[0]
        
        max_indicies = (col, row)

        return max_value, max_indicies
    
    def train(self, subtraction: str = "fixed", ):
        # Apply subtraction
        return


    
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