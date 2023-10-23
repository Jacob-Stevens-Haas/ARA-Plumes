import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes/')
from utils import VideoPointPicker

video_path = input("Please input image path:")

vid_to_click = VideoPointPicker(video_path)
vid_to_click.ask_user()
print(vid_to_click.clicked_point)