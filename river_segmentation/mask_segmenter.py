import cv2

import numpy as np

from .dangerous_segments import DangerousSegments


class MaskSegmentation(DangerousSegments):

    def __init__(self, mask_img):
        self.img = cv2.imread(mask_img)
        grayImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(grayImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        super(MaskSegmentation, self).__init__(hull_list)
