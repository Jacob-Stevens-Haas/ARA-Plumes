import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes')
from utils import *
from tqdm import tqdm

import time
import cv2

class PLUME():
    def __init__(self, video_path):
        self.video_path = video_path
        self.mean_poly = None
        self.var1_poly = None
        self.var2_poly = None
        self.center = None
    

    def extract_frames(self, extension: str = "jpg"):
        video_path = self.video_path


    
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

