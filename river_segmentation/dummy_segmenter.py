import cv2

import numpy as np

from .dangerous_segments import DangerousSegments


class DummySegmentation(DangerousSegments):

    def __init__(self, width=1920, height=1080, x=300):
        self.img = np.zeros((height, width, 3)).astype('uint8')
        self.img[:, x:, :] = 255

        grayImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grayImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        super(DummySegmentation, self).__init__(hull_list)
