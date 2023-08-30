import sys
sys.path.append('/Users/Malachite/Documents/UW/ARA/ARA-Plumes/')
from utils import ImagePointPicker

img_path = input("Please input image path:")

image_to_click = ImagePointPicker(img_path)
image_to_click.ask_user()
print(image_to_click.clicked_point)