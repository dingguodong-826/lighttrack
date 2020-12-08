from __future__ import absolute_import

import numpy as np
import time
from PIL import Image

from ..utils.viz import show_frame
import cv2

class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
    
    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        # times = np.ones((frame_num, )) * 0.001
        times = np.zeros(frame_num)
        boxes[0] = box

        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time
        return boxes, times


from .identity_tracker import IdentityTracker
