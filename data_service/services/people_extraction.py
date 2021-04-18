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
        # print('Extracting')
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(3)]
        frame_data = []
        for img, preds, xyxy in data.get_yolo_results():

            # pose_inpts = np.zeros((len(preds), 3, self.output_dim[0], self.output_dim[1]), dtype=np.uint8)
            pose_list = []

            for person in range(len(preds)):
                cls = preds[person][4]
                conf = preds[person][5]

                if conf > self.conf:
                    continue

                cropped_img = padded_crop_image(img, xyxy[0], padding=15)
                pose_list.append(letterbox_scale(cropped_img, self.output_dim).transpose(2, 0, 1))
                # pose_inpts[person] = letterbox_scale(cropped_img, self.output_dim).transpose(2, 0, 1)
                # torch_img = torch.from_numpy(scaled_img).float()
                # pose_inpts[person] = torch_img/255

                # print('hi')
                # label = f'{cls} {conf:.2f}'
                # plot_one_box(xyxy[person], img, label=label, color=colors[0], line_thickness=3)

            pose_inpts = np.array(pose_list)
            frame_data.append(pose_inpts)
            # inps = torch.zeros(xyxy.size(0), 3, )
            # pt1 = torch.zeros(xyxy.size(0), 2)
            # pt2 = torch.zeros(xyxy.size(0), 2)
            # inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            # inps, pt1, pt2 = crop_from_dets(img, xyxy , inps, pt1, pt2)

            # for *xyxy, conf, cls in reversed(data.results):
            #     label = f'{cls} {conf:.2f}'
            #     plot_one_box(xyxy, img, label=label, color=colors[0], line_thickness=3)

            # cv2.imwrite('./output/{}.png'.format(self.img_counter), img)
            # cv2.imwrite('./output/cropped_{}.png'.format(self.img_counter), img)

            self.img_counter += 1

            # self.img_counter = self.img_counter + 1
        data.add_yolo_post_pro(frame_data)
        # data.preds = [pred.cpu().detach().numpy() for pred in data.preds]
        # data.xyxy = [xyxy.cpu().detach().numpy() for xyxy in data.xyxy]
        self.enqueue(data)
