# import cv2
# import numpy as np

# vid_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/Bryan_videos/Hsv-F0-3000.mp4"
# output_path = "/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/Bryan_videos/new_subtraction.mp4"  # Specify the output file path

# cap = cv2.VideoCapture(vid_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height * 2))  # Double the width and height

# subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=1, detectShadows=False)

# def mean_thresholding(frame, filter_num, ksize_vals, lower_thresh_vals):
#     for i in range(filter_num):
#         ksize = ksize_vals[i]
#         lower_thresh = lower_thresh_vals[i]

#         kernel = np.ones((ksize, ksize), np.float32) / ksize**2
#         frame = cv2.filter2D(frame, -1, kernel)
#         _, frame = cv2.threshold(frame, lower_thresh, 255, cv2.THRESH_BINARY)

#     return frame

# ksize_vals = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3]
# lower_thresh_vals = [230, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 110]

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     mog_frame = subtractor.apply(frame)

#     mask1 = mean_thresholding(mog_frame, 1, ksize_vals, lower_thresh_vals)
#     mask2 = mean_thresholding(mog_frame, 2, ksize_vals, lower_thresh_vals)
#     mask3 = mean_thresholding(mog_frame, 10, ksize_vals, lower_thresh_vals)

#     # Apply Gaussian Blurring
#     kernel_size = (71, 41)
#     sigma = 30
#     sigma_x = sigma
#     sigma_y = sigma

#     # Convert grayscale images to color (BGR)
#     mog_frame_color = cv2.cvtColor(mog_frame, cv2.COLOR_GRAY2BGR)
#     mask1_color = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
#     mask2_color = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
#     mask3_color = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)

#     mask3 = cv2.GaussianBlur(mask3, kernel_size, sigma_x, sigma_y)

#     mask3_color_2 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)

#     # Find contours in mask3
#     contours, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create blank array to fill in with contours
#     filled_mask = np.zeros_like(mask3_color)

#     # Color to be used on contours
#     fill_color = (0, 0, 255)

#     n = 1
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     selected_contours = contours[:n]

#     for contour in selected_contours:
#         cv2.drawContours(filled_mask, [contour], 0, fill_color, thickness=2)

#     # Create 2x2 grid
#     top_row = np.hstack((mog_frame_color, mask3_color))
#     bottom_row = np.hstack((mask3_color_2, filled_mask))
#     display_frame = np.vstack((top_row, bottom_row))

#     # Write the frame to the output video
#     out.write(display_frame)

#     cv2.imshow("Mask", display_frame)
#     key = cv2.waitKey(fps)
#     if key == 27:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Function to perform background subtraction and save the video as MP4
def background_subtraction_and_save_as_mp4(video_path, n_frames, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read the first frame to initialize the background
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Convert the first frame to grayscale for background modeling
    average_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Get the frame dimensions and create a VideoWriter object
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video as MP4
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height), isColor=False)

    frame_count = 1  # Initialize frame_count before the loop

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Accumulate the first n frames
        if frame_count <= n_frames:
            frame_count += 1
            cv2.accumulate(frame_gray, average_frame)
            if frame_count == n_frames:
                average_frame /= n_frames

        # Perform background subtraction for all frames after the first n frames
        else:
            frame_diff = cv2.subtract(frame_gray, average_frame.astype(np.uint8))
            out.write(frame_diff)

        # Press 'q' to exit the video playback
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture, VideoWriter, and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = '/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/Bryan_videos/Hsv-F0-3000.mp4'  # Replace with your video file path
    n_frames = 5  # Number of frames to use for background averaging
    output_video_path = '/Users/Malachite/Documents/UW/ARA/ARA-Plumes/plume_videos/Bryan_videos/fix_sub.mp4'  # Output video file path with .mp4 extension

    background_subtraction_and_save_as_mp4(video_file, n_frames, output_video_path)
