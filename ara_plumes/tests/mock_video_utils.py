from unittest.mock import MagicMock

import cv2
import numpy as np


# Mock video for testing
class MockVideoCapture(MagicMock):
    def __init__(self, frame_count=100, frame_width=640, frame_height=480):
        super().__init__()
        self.frame_count = frame_count
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.current_frame = 0

        # Pre-generate frames with varying intensity
        self.frames = [
            np.full((self.frame_height, self.frame_width, 3), i % 255, dtype=np.uint8)
            for i in range(self.frame_count)
        ]

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return super().get(prop_id)

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self.current_frame = value

    def read(self):
        if self.current_frame < self.frame_count:
            frame = self.frames[self.current_frame]
            self.current_frame += 1
            return True, frame
        return False, None
