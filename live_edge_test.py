# import cv2
# import numpy as np

# # Enable webcam to capture video
# cap = cv2.VideoCapture(0)

# # Set the desired frame size
# frame_width = 640
# frame_height = 480

# # Get time and frames, make capture continuous
# while True:
#     # Read frames from camera
#     ret, frame = cap.read()

#     if ret:
#         # Resize frame to desired dimensions
#         frame = cv2.resize(frame, (frame_width, frame_height))

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply bilateral filtering
#         gray_filtered = cv2.bilateralFilter(gray, d=30, sigmaColor=30, sigmaSpace=20)

#         # Detect edges with different Canny parameters for each video
        # edges1 = cv2.Canny(frame, 50, 75)
        # edges2 = cv2.Canny(frame, 75, 100)
        # edges3 = cv2.Canny(frame, 100, 125)
        # edges4 = cv2.Canny(frame, 125, 150)

#         # Create a blank canvas to arrange videos
#         canvas = np.zeros((frame_height * 2, frame_width * 2), dtype=np.uint8)

#         # Arrange videos in a 2x2 grid on the canvas
#         canvas[:frame_height, :frame_width] = edges1
#         canvas[:frame_height, frame_width:] = edges2
#         canvas[frame_height:, :frame_width] = edges3
#         canvas[frame_height:, frame_width:] = edges4

#         # Show the canvas with all four videos
#         cv2.imshow("Videos", canvas)

#         # Assign kill key as Esc (27)
#         if cv2.waitKey(1) == 27:
#             break

# # Close window
# cap.release()
# cv2.destroyAllWindows()

####################################################################################
# import cv2
# import numpy as np

# # Enable webcam to capture video
# cap = cv2.VideoCapture(0)

# # Set the desired canvas size
# canvas_width = 1280
# canvas_height = 960

# # Get time and frames, make capture continuous
# while True:
#     # Read frames from camera
#     ret, frame = cap.read()

#     if ret:
#         # Resize frame to match the desired canvas size
#         frame = cv2.resize(frame, (canvas_width // 2, canvas_height // 2))

#         # Get the resized frame size
#         frame_height, frame_width = frame.shape[:2]

#         # Create a blank canvas
#         canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

#         # Calculate the starting positions to center the video frames on the canvas
#         start_x = (canvas_width - frame_width * 2) // 2
#         start_y = (canvas_height - frame_height * 2) // 2

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply bilateral filtering
#         gray_filtered = cv2.bilateralFilter(gray, d=30, sigmaColor=30, sigmaSpace=20)

#         # Detect edges with different Canny parameters for each video
#         edges1 = cv2.Canny(frame, 50, 75)
#         edges2 = cv2.Canny(frame, 75, 100)
#         edges3 = cv2.Canny(frame, 100, 125)
#         edges4 = cv2.Canny(frame, 125, 150)

#         # Convert grayscale images to 3-channel images
#         edges1 = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)
#         edges2 = cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR)
#         edges3 = cv2.cvtColor(edges3, cv2.COLOR_GRAY2BGR)
#         edges4 = cv2.cvtColor(edges4, cv2.COLOR_GRAY2BGR)

#         # Arrange videos on the canvas
#         canvas[start_y:start_y + frame_height, start_x:start_x + frame_width] = edges1
#         canvas[start_y:start_y + frame_height, start_x + frame_width:start_x + frame_width * 2] = edges2
#         canvas[start_y + frame_height:start_y + frame_height * 2, start_x:start_x + frame_width] = edges3
#         canvas[start_y + frame_height:start_y + frame_height * 2, start_x + frame_width:start_x + frame_width * 2] = edges4

#         # Show the canvas with all four videos
#         cv2.imshow("Videos", canvas)

#         # Assign kill key as Esc (27)
#         if cv2.waitKey(1) == 27:
#             break

# # Close window
# cap.release()
# cv2.destroyAllWindows()

####################################################################################


# import cv2
# import numpy as np

# # Enable webcam to capture video
# cap = cv2.VideoCapture(0)

# # Set the desired canvas size
# canvas_width = 1280
# canvas_height = 960

# # Get time and frames, make capture continuous
# while True:
#     # Read frames from camera
#     ret, frame = cap.read()

#     if ret:
#         # Resize frame to match the desired canvas size
#         frame = cv2.resize(frame, (canvas_width // 2, canvas_height // 2))

#         # Get the resized frame size
#         frame_height, frame_width = frame.shape[:2]

#         # Create a blank canvas
#         canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

#         # Calculate the starting positions to center the video frames on the canvas
#         start_x = (canvas_width - frame_width * 2) // 2
#         start_y = (canvas_height - frame_height * 2) // 2

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply bilateral filtering
#         gray_filtered = cv2.bilateralFilter(gray, d=30, sigmaColor=30, sigmaSpace=20)

#         # Detect edges with different Canny parameters for each video
#         edges1 = cv2.Canny(frame, 50, 75)
#         edges2 = cv2.Canny(frame, 75, 100)
#         edges3 = cv2.Canny(frame, 100, 125)
#         edges4 = cv2.Canny(frame, 125, 150)

#         # Arrange videos on the canvas
#         canvas[start_y:start_y + frame_height, start_x:start_x + frame_width] = edges1
#         canvas[start_y:start_y + frame_height, start_x + frame_width:start_x + frame_width * 2] = edges2
#         canvas[start_y + frame_height:start_y + frame_height * 2, start_x:start_x + frame_width] = edges3
#         canvas[start_y + frame_height:start_y + frame_height * 2, start_x + frame_width:start_x + frame_width * 2] = edges4

#         # Show the canvas with all four videos
#         cv2.imshow("Videos", canvas)

#         # Assign kill key as Esc (27)
#         if cv2.waitKey(1) == 27:
#             break

# # Close window
# cap.release()
# cv2.destroyAllWindows()

##################################################################################################
import cv2
import numpy as np

# Enable webcam to capture video
cap = cv2.VideoCapture(0)

# Set the desired canvas size
canvas_width = 1280
canvas_height = 960

# Get time and frames, make capture continuous
while True:
    # Read frames from camera
    ret, frame = cap.read()

    if ret:
        # Resize frame to match the desired canvas size
        frame = cv2.resize(frame, (canvas_width // 2, canvas_height // 2))

        # Get the resized frame size
        frame_height, frame_width = frame.shape[:2]

        # Create a blank canvas
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        # Calculate the starting positions to center the video frames on the canvas
        start_x = (canvas_width - frame_width * 2) // 2
        start_y = (canvas_height - frame_height * 2) // 2

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filtering to grayscale frame
        gray_filtered = cv2.bilateralFilter(gray_frame, d=30, sigmaColor=30, sigmaSpace=20)

        # Detect edges with different Canny parameters for each video
        edges1 = gray_frame
        edges2 = cv2.Canny(gray_filtered, 0,25 )
        edges3 = cv2.Canny(gray_filtered, 25, 50)
        edges4 = cv2.Canny(gray_filtered, 50, 75)

        # Arrange videos on the canvas
        canvas[start_y:start_y + frame_height, start_x:start_x + frame_width] = edges1
        canvas[start_y:start_y + frame_height, start_x + frame_width:start_x + frame_width * 2] = edges2
        canvas[start_y + frame_height:start_y + frame_height * 2, start_x:start_x + frame_width] = edges3
        canvas[start_y + frame_height:start_y + frame_height * 2, start_x + frame_width:start_x + frame_width * 2] = edges4

        # Show the canvas with all four videos
        cv2.imshow("Videos", canvas)

        # Assign kill key as Esc (27)
        if cv2.waitKey(1) == 27:
            break

# Close window
cap.release()
cv2.destroyAllWindows()
