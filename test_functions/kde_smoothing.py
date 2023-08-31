#############################################
## To test KDE smoothing in space and time ##
#############################################
import cv2
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes')
from utils import create_directory, create_id

# Test smoothing July 20 video_high_1 subtracted frames
img_folder_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/fixed_avg_frames/"
img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/fixed_avg_frames/subtract_0256.png"
img = cv2.imread(img_path)

######################
## Helper functions ##
######################

def get_img_paths(directory: str,
                  extension: str ="png"):
    png_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.'+extension):
                png_paths.append(os.path.join(root, file))
    png_paths.sort()
    return png_paths

############################
## Gaussian Blur in Space ##
############################
blur_in_space = False

if blur_in_space == True:
    # Apply Guassian blurring
    kernel_size = (25,25)
    sigma = 4
    sigma_x = sigma
    sigma_y = sigma

    blurred_img = cv2.GaussianBlur(img, kernel_size,sigma_x,sigma_y)

    cv2.imshow("orig", img)
    cv2.imshow("Gaussian blur", blurred_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussian_space_blur(directory,
                        kernel_size,
                        sigma,
                        save_directory,
                        extension="png"):

    create_directory(save_directory)

    img_paths = get_img_paths(directory)

    frame_mag = len(str(len(img_paths)))
    i=0
    for img in tqdm(img_paths):
        # Apply blurring
        img = cv2.imread(img)
        blurred_img = cv2.GaussianBlur(img,kernel_size,sigma,sigma)
        
        # Save img
        new_id = create_id(i,frame_mag)
        file_name = "gauss_blur_"+new_id+"."+extension
        cv2.imwrite(os.path.join(save_directory,file_name),blurred_img)
        i+=1

    return

###########################
## Gaussian Blur in Time ##
###########################
def average_weighted_images(imgs,
                            weights):
    
    a = np.array(imgs)
    b = np.array(weights)[:,np.newaxis,np.newaxis,np.newaxis] # Might need to change if working with gray images
    c = np.round(np.sum(a*b,axis=0)).astype(int)
    return c


def gaussian_time_blur(directory: list,
                       window,
                       sigma,
                       save_directory,
                       extension="png"):
    # check that window is positive odd int.                
    if not isinstance(window,int):
        raise Exception("window must be a positive odd int.")
    if not window%2==1:
        raise Exception("window must be a positive odd int.")

    create_directory(save_directory)

    # Get images directory paths in list  
    img_paths = get_img_paths(directory)
    frame_mag = len(str(len(img_paths)))
    print("frame_mag:", frame_mag)

    
    # Create Gaussian Weights
    gauss_center = window//2
    weights = np.zeros(window)
    for x in range(window):
        weights[x] = np.exp(-(x-gauss_center)**2 / (2*sigma**2))
    weights /= np.sum(weights)

    blur_id = 0
    for i in tqdm(range(len(img_paths)-window+1)):
        imgs = [cv2.imread(img) for img in img_paths[i:window+i]]
        avg_img = average_weighted_images(imgs,weights)
        avg_img = cv2.convertScaleAbs(avg_img) # Conver to 8bit datatype

        # Save image
        new_id = create_id(blur_id,frame_mag)
        # print("new_id:", new_id)
        file_name = "gauss_time_blur_"+new_id+"."+extension
        # print("file_name:", file_name)
        # print("full_name:", os.path.join(save_directory,file_name))
        cv2.imwrite(os.path.join(save_directory,file_name), avg_img)
        blur_id+=1
    return weights


window=11
sigma=5

##################################
## Apply kde smoothing in space ##
##################################
save_directory = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/space_blur"
kernel_size = (25,25)
sigma=4

# gaussian_space_blur(directory=img_folder_path,
#                     kernel_size=kernel_size,
#                     sigma=sigma,
#                     save_directory=save_directory)

#################################
## Apply kde smoothing in time ##
#################################

directory = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/space_blur"
save_directory = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/time_blur"
window=5
sigma=2
# 11 and 5
gaussian_time_blur(directory=directory, window=window,sigma=sigma,save_directory=save_directory)

#########################
## Apply path learning ##
######################

# img_path = get_img_paths(img_folder_path)
# frame_mag = len(str(len(img_path)))
# print(frame_mag)

# for i in range(10):
#     new_id = create_id(i,frame_mag)
#     file_name="gauss_time_blur_"+new_id+"."+"png"
#     full_name = os.path.join(save_directory,file_name)
#     print(full_name)

# For testing values
# Create Gaussian Weights
# gauss_center = window//2
# weights = np.zeros(window)
# for x in range(window):
#     weights[x] = np.exp(-(x-gauss_center)**2 / (2*sigma**2))
# weights /= np.sum(weights)

# print("Weights:", weights)
# print("Weight sum:", sum(weights))
# img_path = get_img_paths(img_folder_path)
# imgs = [cv2.imread(path) for path in img_path[218:218+window]]


# avg_img = average_weighted_images(imgs,weights)
# avg_img=cv2.convertScaleAbs(avg_img) # To convert to the correct type for cv2


# for i,img in enumerate(imgs):
#     cv2.imshow(f"Img {i+1}",img)

# cv2.imshow("avg Img", avg_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





