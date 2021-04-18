import cv2
import time
import torch
import queue

import numpy as np
import multiprocessing as mp

from flask import Flask, Response
from data_utils.torch_utils import img_to_torch

from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

from data_service.data_service import DataService
from data_utils.datatypes import SimpleFIFOBuffer
from data_models import FrameBuffer, ExtractedPeopleResults

from yolov5_master.utils.plots import plot_one_box

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WebServerOutput(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, frame_rate):
        super(WebServerOutput, self).__init__(WebServerOutput.input_type, WebServerOutput.output_type)
        self.frame_rate = frame_rate
        self.frame_time = float(1)/self.frame_rate
        self.q = mp.Queue(maxsize=90)
        # self.q = None
        self.last_image = np.zeros((720, 1080, 3))

    def _data_ingest(self, data):

        imgs = data.get_drown_detection_images()

        for i in range(len(imgs)):
            if self.q.full():
                # break
                self.q.get()
            self.q.put(imgs[i])

        print('Compute Time: {}s'.format(time.time() - data.enqueue_time))

    def clear_buffer(self):
        self.q = mp.Queue(maxsize=1000)

    def get_frame(self):
        print(self.q.qsize())
        time.sleep(0.01)

        while self.q.empty():
            # print('Spinning')
            return self.last_image

        return self.q.get()
