import cv2
import numpy as np

vid_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/Bryan_videos/Hsv-F0-3000.mp4"
cap = cv2.VideoCapture(vid_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=1, detectShadows=False)

def mean_thresholding(frame, filter_num, ksize_vals, lower_thresh_vals):
    for i in range(filter_num):
        ksize = ksize_vals[i]
        lower_thresh = lower_thresh_vals[i]

        kernel = np.ones((ksize, ksize), np.float32) / ksize**2
        frame = cv2.filter2D(frame, -1, kernel)
        _, frame = cv2.threshold(frame, lower_thresh, 255, cv2.THRESH_BINARY)

    return frame

ksize_vals = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3]
lower_thresh_vals = [230, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 110]

while True:
    ret, frame = cap.read()

    mog_frame = subtractor.apply(frame)

    mask1 = mean_thresholding(mog_frame, 1, ksize_vals, lower_thresh_vals)
    mask2 = mean_thresholding(mog_frame, 2, ksize_vals, lower_thresh_vals)
    mask3 = mean_thresholding(mog_frame, 10, ksize_vals, lower_thresh_vals)

    # Apply Gaussian Blurring
    kernel_size =(71,41)
    sigma=20
    sigma_x = sigma
    sigma_y = sigma

    

    # Convert grayscale images to color (BGR)
    mog_frame_color = cv2.cvtColor(mog_frame, cv2.COLOR_GRAY2BGR)
    mask1_color = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
    mask2_color = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    mask3_color = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)

    mask3 = cv2.GaussianBlur(mask3, kernel_size, sigma_x, sigma_y)

    mask3_color_2 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)

    # Find contours in mask3
    contours, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank array to fill in with contours
    filled_mask = np.zeros_like(mask3_color)

    # Color to be used on contours
    fill_color = (0, 255, 0)

    # combined_contour = []
    # for contour in contours:
    #     combined_contour.extend(contour)

    # # print(len(combined_contour))
    # if combined_contour:
    #     # print("test")
    #     combined_contour = np.array(combined_contour)
    #     cv2.drawContours(filled_mask, [combined_contour],-1,fill_color, thickness=cv2.FILLED)
    for contour in contours:
        cv2.drawContours(filled_mask, [contour], 0, fill_color, thickness=cv2.FILLED)

    contours, _ = cv2.findContours(cv2.cvtColor(filled_mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(filled_mask,[contour],0,(0,0,255), thickness=2)




    # Create 2x2 grid
    top_row = np.hstack((mog_frame_color, mask3_color))
    bottom_row = np.hstack((mask3_color_2, filled_mask))
    display_frame = np.vstack((top_row, bottom_row))

    cv2.imshow("Mask", display_frame)
    key = cv2.waitKey(fps)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


   # cv2.imshow("Mog Frame", mog_frame)
    # cv2.imshow("Mask_1", mask1)   
    # cv2.imshow("Mask_2", mask3)
    # cv2.imshow("Mask_3", mask3)


# subtractor2 = cv2.createBackgroundSubtractorMOG2(varThreshold=1, detectShadows=False)
# subtractor3 = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)

# Apply median (Alex h)
# cv2.medianBlur()
# cv2.fastNlMeanDenoising
# 180 threshold value

# Apply moving window average 
# Fill in images with "graph alg - flood fill, cut alg"
# or superimpose images?
# py_grabcut


    # mask1 = subtractor1.apply(frame)
 
    # mask2 = subtractor2.apply(frame)
    # ksize = 5
    # kernel = np.ones((ksize,ksize),np.float32)/ksize**2
    # mask1 = cv2.filter2D(mask2,-1,kernel)
    # _, mask4 = cv2.threshold(mask1, 230, 255, cv2.THRESH_BINARY)

    # ksize = 3
    # kernel = np.ones((ksize,ksize),np.float32)/ksize**2
    # mask2 = cv2.filter2D(mask4, -1, kernel)
    # _, mask3 = cv2.threshold(mask2, 230, 255, cv2.THRESH_BINARY)

    # ksize = 3
    # kernel = np.ones((ksize,ksize),np.float32)/ksize**2
    # mask2 = cv2.filter2D(mask3, -1, kernel)
    # _, mask2 = cv2.threshold(mask2, 150, 255, cv2.THRESH_BINARY)

    # mask3 = subtractor3.apply(mask2)
    # mask2 = cv2.medianBlur(mask2, 3)