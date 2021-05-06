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

from data_service.data_output import DataOutput
from data_utils.datatypes import SimpleFIFOBuffer
from data_models import FrameBuffer, ExtractedPeopleResults

from yolov5_master.utils.plots import plot_one_box

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WebServerOutput(DataOutput):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, frame_rate, FPS=30, img_dim=(640, 480)):
        super(WebServerOutput, self).__init__(WebServerOutput.input_type)
        self.frame_rate = frame_rate
        self.frame_time = float(1)/self.frame_rate
        self.q = mp.Queue(maxsize=90)
        self.last_image = np.zeros((480, 720, 3))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.video_writer = cv2.VideoWriter('./output/video0.avi', fourcc, FPS, (img_dim[0], img_dim[1]))
        self.video_reader = None

    def _data_ingest(self, data):

        imgs = data.get_drown_detection_images()

        for i in range(len(imgs)):
            self.video_writer.write(imgs[i])

        print('Compute Time: {}s'.format(time.time() - data.enqueue_time))

    def release_video(self):
        self.video_writer.release()

    def clear_buffer(self):
        self.q = mp.Queue(maxsize=1000)

    def get_frame(self):

        if self.video_reader is None:
            return self.last_image

        ret, frame = self.video_reader.read()

        return frame
