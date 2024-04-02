import glob
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm


##################
# Math Functions #
##################
def circle_intersection(x0, y0, r0, x1, y1, r1):
    """
    Find the intersections of a circle centered at (x0,y0) with radii
    r0 with a circle centered at (x1,y1) with radii r1.

    source:
    https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect
    """
    d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    L = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(r0**2 - L**2)

    x3 = L / d * (x1 - x0) - h / d * (y1 - y0) + x0
    y3 = L / d * (y1 - y0) + h / d * (x1 - x0) + y0

    x4 = L / d * (x1 - x0) + h / d * (y1 - y0) + x0
    y4 = L / d * (y1 - y0) - h / d * (x1 - x0) + y0

    sol = np.array([[x3, y3], [x4, y4]])

    return sol


def circle_poly_intersection(r, x0, y0, a, b, c, real=True):
    """
    Find the intersection point(s) of a circle centered at (x0,y0) with radius
    r and a polynomial of the form y=ax^2+bx+c.

    Find roots of the poly g(x) = x^2 + (ax^2+bx+c)^2-r^2
    """

    coeff = [
        c**2 - 2 * c * y0 + y0**2 + x0**2 - r**2,
        2 * b * c - 2 * b * y0 - 2 * x0,
        1 + b**2 + 2 * a * c - 2 * a * y0,
        2 * a * b,
        a**2,
    ]
    roots = np.polynomial.polynomial.polyroots(coeff)

    if real is True:
        roots = np.real(roots[np.isreal(roots)])

    def f_poly(x):
        return a * x**2 + b * x + c

    # y = f_poly(roots)

    sol = []
    for x in roots:
        y = f_poly(x)
        sol.append([x, y])

    sol = np.array(sol)
    return sol


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
    for png_file in tqdm(png_files):
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
    """
    Create movie directly form vari_dist data attained from
    PLUME.train(). In PLUME.var1_dist and PLUME.var2_dist.
    """

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


def create_ROM_plume_movie(
    PLUME_object,
    frame_range: int | list[int] | None=None,
    save_path: str | None=None
) -> None:
    """
    Create the ROM plume movie using the trained PLUME model
    """

    if frame_range is None:
        num_frames = len(PLUME_object.mean_poly)
    elif isinstance(frame_range, int):
        num_frames = range(frame_range, len(PLUME_object.mean_poly))
    elif isinstance(frame_range, list):
        num_frames = range(frame_range[0], frame_range[1])

    def generate_plot(frame):
        PLUME_object.plot_ROM_plume(frame, show_plot=False)

    fig = plt.figure()
    ani = FuncAnimation(fig, generate_plot, frames=num_frames)

    if isinstance(save_path, str):
        if save_path.endswith(".mp4") is False:
            save_path += ".mp4"
        ani.save(save_path, writer="ffmpeg", fps=15)
        plt.show()
