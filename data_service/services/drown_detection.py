import cv2
import time
import torch

import numpy as np

from data_utils.torch_utils import img_to_torch

from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

from data_service.data_service import DataService
from data_utils.datatypes import SimpleFIFOBuffer
from data_models import FrameBuffer, ExtractedPeopleResults

from yolov5_master.utils.plots import plot_one_box

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DrownDetection(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, river_segmentation_map):
        super(DrownDetection, self).__init__(DrownDetection.input_type, DrownDetection.output_type)
        self.img_counter = 0
        self.river_segmentation_map = river_segmentation_map
        self.drown_risk = {}

    def _data_ingest(self, data):

        seen_ids = []

        contours = self.river_segmentation_map.get_segments()

        final_imgs = []

        for img, xyxy, tracked in data.get_tracked_frames():

            for contour in contours:
                # cv2.fillConvexPoly(img, contour, color=(0, 0, 0))
                cv2.drawContours(img, [contour], 0, color=(255, 0, 0), thickness=10)

            for idx in range(len(tracked)):

                pos_history = tracked[idx].get_pos_history()

                for p in pos_history:
                    if np.any(np.isnan(p)):
                        continue
                    cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 0, 0), markerType=cv2.MARKER_CROSS,
                                   markerSize=30, thickness=2, line_type=cv2.LINE_AA)

                if len(pos_history) < 10:
                    continue

                seen_ids.append(tracked[idx].id)
                if tracked[idx].id in self.drown_risk:
                    person = self.drown_risk[tracked[idx].id]
                else:
                    person = DrownRiskProfile(tracked[idx].id)
                    self.drown_risk[tracked[idx].id] = person

                pos_history = tracked[idx].get_pos_history()
                # diff = np.floor(len(pos_history)/5).astype(np.int)
                # sub_sampled = pos_history[0::diff]
                sub_sampled = pos_history[-25::2]

                if np.any(np.isnan(sub_sampled)): continue

                fit = np.polyfit(sub_sampled[:, 0], sub_sampled[:, 1], 1)
                prediction = np.poly1d(fit)

                direction = pos_history[-1, 0] - pos_history[-10, 0]

                new_pts = np.array([pos_history[-1, 0] + (direction * i) for i in range(6)])

                new_ys = prediction(new_pts)

                pts = np.vstack((new_pts, new_ys)).T

                if self.river_segmentation_map.check_in_segment(pts):
                    person.add_predicted_to_fall_in()

                if self.river_segmentation_map.check_in_segment([pos_history[-1]]):
                    person.inside_river_segmentation()

                plot_one_box(xyxy[idx], img, label=str(person.tracker_id), color=colours[person.risk], line_thickness=3)

                for p_x, p_y in zip(new_pts, new_ys):
                    cv2.drawMarker(img, (int(p_x), int(p_y)), color=colours[person.risk], markerType=cv2.MARKER_CROSS,
                                   markerSize=30, thickness=2, line_type=cv2.LINE_AA)

            final_imgs.append(img)
            self.img_counter += 1
        data.add_drown_detection_images(final_imgs)
        self.enqueue(data)
        # print('Compute Time here: {}s'.format(time.time() - data.enqueue_time))


class DrownRiskProfile:

    LOW = 0
    MEDIUM = 1
    HIGH = 2

    def __init__(self, id):
        self.tracker_id = id
        self.risk = DrownRiskProfile.LOW
        self.warnings = 0

    def add_predicted_to_fall_in(self):
        self.warnings += 1
        self.risk = DrownRiskProfile.MEDIUM

    def inside_river_segmentation(self):
        self.risk = DrownRiskProfile.HIGH



colours = {
    DrownRiskProfile.HIGH: (255, 0, 0),
    DrownRiskProfile.MEDIUM: (255, 191, 0),
    DrownRiskProfile.LOW: (0, 255, 0),
}