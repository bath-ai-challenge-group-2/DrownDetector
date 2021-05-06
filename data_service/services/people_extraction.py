import cv2
import time
import torch
import random

import numpy as np
from data_utils.torch_utils import img_to_torch

from data_service.data_service import DataService
from data_models import ExtractedPeopleResults

from data_utils.img_utils import padded_crop_image, letterbox_scale
from yolov5_master.utils.plots import plot_one_box

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PeopleExtraction(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    # def __init__(self, conf=0.75, output_dim=(320, 256)):
    def __init__(self, conf=0.75, output_dim=(160, 128)):
        super(PeopleExtraction, self).__init__(PeopleExtraction.input_type, PeopleExtraction.output_type)
        self.conf = conf
        self.output_dim = np.array(output_dim)
        self.img_counter = 0

    def _data_ingest(self, data):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(3)]
        frame_data = []
        for img, preds, xyxy in data.get_yolo_results():

            pose_list = []

            for person in range(len(preds)):
                cls = preds[person][4]
                conf = preds[person][5]

                if conf > self.conf:
                    continue

                cropped_img = padded_crop_image(img, xyxy[0], padding=15)
                pose_list.append(letterbox_scale(cropped_img, self.output_dim).transpose(2, 0, 1))

            pose_inpts = np.array(pose_list)
            frame_data.append(pose_inpts)

            self.img_counter += 1

        data.add_yolo_post_pro(frame_data)
        self.enqueue(data)
