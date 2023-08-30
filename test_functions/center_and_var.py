# Script for testing variance detection through the use of cv2.findContours

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes/')

from utils import ImagePointPicker, learn_center_var

img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0287.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0102.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0690.png"
# img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_low_1/fixed_avg_frames/subtract_0628.png"

img_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/July_20/video_high_1/fixed_avg_frames/subtract_0005.png"


#########################
## Ask user for center ##
#########################

image_to_click = ImagePointPicker(img_path)
image_to_click.ask_user()
orig_center = image_to_click.clicked_point 

print("orig center:", type(orig_center), orig_center)

eps_smooth = 10
print("testing unsmoothed")
contour_image_original,_,_,_ = learn_center_var(img_path=img_path,
                                          orig_center=orig_center)
print("success\n")

print("testing smoothed")
contour_image_smoothed,_,_,_ = learn_center_var(img_path=img_path,
                                          orig_center=orig_center,
                                          smoothing=True,
                                          eps_smooth=eps_smooth)
print("success")

# Display the images side by side
# print(contour_image_original.shape)
# print(contour_image_smoothed.shape)
# print()
comparison_image = cv2.hconcat([contour_image_original, contour_image_smoothed])

# Display the image with contours
cv2.imshow(f"Orig vs eps={eps_smooth} smoothed Contours", comparison_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

