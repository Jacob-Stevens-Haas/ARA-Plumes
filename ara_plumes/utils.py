import glob
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip
from PIL import Image
from scipy.optimize import curve_fit
from tqdm import tqdm


##################
# For Regression #
##################


def flatten_vari_dist(vari_dist):
    """
    Convert vari_dist list [(t0,[[x0,y0],...[xn,yn]]),...]
    to flattned array ->[[t0,x0,y0], [t0,x1,y1],...[t0,xn,yn],[t1,...],...]
    """
    t_x_y = []
    for vari in vari_dist:
        nt = len(vari[1])
        ti = vari[0]
        if nt == 0:
            continue
        t_x_y_i = np.array([ti for _ in range(nt)]).reshape(nt, -1)
        t_x_y_i = np.hstack((t_x_y_i, vari[1]))
        if len(t_x_y) == 0:
            t_x_y = t_x_y_i
        else:
            t_x_y = np.vstack((t_x_y, t_x_y_i))
    return t_x_y


def plot_sinusoid(X_i, Y_i, t_i, regress=True, initial_guess=(1, 1, 1, 1)):
    fig = plt.figure(figsize=(8, 6))
    if regress is True:
        try:
            A, w, gamma, B = sinusoid_regression(X_i, Y_i, t_i, initial_guess)
            x = np.linspace(0, X_i[-1])

            def sinusoid_func(x):
                return A * np.sin(w * x - gamma * t_i) + B * x

            y = sinusoid_func(x)

            plt.plot(x, y, color="red", label="sinusoid")
            new_guess = (A, w, gamma, B)
        except Exception as e:
            print(f"Sinusoid could not fit. Error: {e}")
            new_guess = initial_guess

    plt.scatter(X_i, Y_i, color="blue", label="var points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sinusoid, t=" + str(t_i))
    plt.legend()
    # plt.grid(True)
    # plt.show(block=False)
    return fig, new_guess


def sinusoid_regression(X, Y, t, initial_guess):

    # Define the function
    def sinusoid(x, A, w, gamma, B):
        return A * np.sin(w * x - gamma * t) + B * x

    # initial_guess = (1, 1, 1, 1)
    params, covariance = curve_fit(sinusoid, X, Y, initial_guess)
    A_opt, w_opt, gamma_opt, B_opt = params
    return (A_opt, w_opt, gamma_opt, B_opt)


#############################
# General Purpose functions #
#############################

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
        if filename.endswith("." + extension):
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
            frame_str = ""
            for i in range(num_of_lead_zeros):
                frame_str += "0"
            frame_number = frame_str + str(frame_number)

        if not isinstance(frame_number, str):
            frame_number = str(frame_number)

        frame_path = f"{save_path}/frame_" + frame_number + "." + extension
        imageio.imwrite(frame_path, frame)


# Functions for subtracting frames
def create_id(id, magnitude):
    num_of_leading_zeros = magnitude - len(str(id))
    frame_str = ""
    for i in range(num_of_leading_zeros):
        frame_str += "0"
    frame_number = frame_str + str(id)
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
        if filename.endswith("." + extension):
            file_ids.append(filename)
            # print(file_ids)
    file_ids.sort()
    return file_ids


############################
# Edge detection functions #
############################


# ADD variables for hyperparameter tuning
def edge_detect(
    frames_path,
    extension="png",
    save_path="edge_frames",
    d=10,
    sigmaColor=20,
    sigmaSpace=20,
    t_lower=5,
    t_upper=10,
):
    """
    Uses Bilaterial filtering in conjunction with opencv edge detection.
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
        gray_filtered = cv2.bilateralFilter(
            gray, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
        )

        # Use Canny to get edge contours
        edges_filtered = cv2.Canny(
            gray_filtered, threshold1=t_lower, threshold2=t_upper
        )

        # save edge detected frame
        new_id = create_id(count, frames_mag)
        file_name = "edge_" + new_id + "." + extension
        cv2.imwrite(os.path.join(save_path, file_name), edges_filtered)

        count += 1
    return


######################
# Finding Plume Path #
######################

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


class VideoPointPicker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.clicked_point = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked at ({x}, {y})")
            cv2.destroyAllWindows()

    def ask_user(self):
        video_cap = self.video_capture
        tot_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_id = tot_frames // 2
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_id)

        ret, frame = video_cap.read()

        if ret:
            cv2.namedWindow("Image")
            cv2.setMouseCallback("Image", self.mouse_callback)
            cv2.imshow("Image", frame)

            while self.clicked_point is None:
                cv2.waitKey(1)


###################
# Post Processing #
###################

# For saving frames as video
def create_video(directory, output_file, fps=15, extension="png", folder="movies"):
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
    png_files = sorted(
        [file for file in os.listdir(directory) if file.endswith("." + extension)]
    )

    # Get the first image to retrieve its dimensions
    first_image = cv2.imread(os.path.join(directory, png_files[0]))
    height, width, _ = first_image.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each image to the video writer
    for png_file in png_files:
        image_path = os.path.join(directory, png_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the video writer and close the file
    video_writer.release()


def create_gif(
    frames_dir: str,
    duration: int,
    frames_range: list = None,
    rotate_ang: float = 0,
    gif_name: str = "animation",
    extension: str = "png",
):
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
        if not string.endswith("/"):
            string += "/"
        return string

    # start_frame, end_frame = frames_range

    # Add '/' to path if it does not already exist
    frames_dir = add_forward_slash(frames_dir)

    # Get all files with specified extension in frames_dir
    frame_paths = sorted(glob.glob(frames_dir + "*." + extension))

    # Select appropriate frame range - default None is all frames
    if frames_range is None:
        frame_paths = frame_paths
    elif isinstance(frames_range, list) and len(frames_range) == 2:
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
    frames[0].save(
        gif_name + ".gif",
        format="GIF",
        append_images=frames[1:],
        duration=duration,
        save_all=True,
        loop=0,
    )


def create_vari_dist_movie(vari_dist, save_path=None):

    # find max r and d
    max_r = 0
    max_d = 0
    for vari in vari_dist:
        if len(vari[1]) != 0:
            max_r_i = np.max(vari[1][:, 0])
            if max_r_i >= max_r:
                max_r = max_r_i
            max_d_i = np.max(vari[1][:, 1])
            if max_d_i >= max_d:
                max_d = max_d_i

    # function to generate plots
    def generate_plot(frame):
        plt.clf()
        t, r_d_arr = vari_dist[frame]
        plt.title(f"Var (r,d), t={t}")
        if len(r_d_arr) != 0:
            plt.scatter(r_d_arr[:, 0], r_d_arr[:, 1], c="blue")
        plt.xlim(0, max_r)
        plt.ylim(0, max_d)
        plt.xlabel("r (flattened p_mean)")
        plt.ylabel("d")

    # Create movie
    fig = plt.figure()
    ani = FuncAnimation(fig, generate_plot, frames=len(vari_dist), interval=100)
    if isinstance(save_path, str):
        ani.save(save_path, writer="ffmpeg", fps=10)
        plt.show()
