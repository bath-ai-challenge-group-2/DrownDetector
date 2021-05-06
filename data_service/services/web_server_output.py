import cv2
import time
import torch
import queue

import numpy as np
import multiprocessing as mp

from data_service.data_service import DataService
from data_utils.datatypes import SimpleFIFOBuffer
from data_models import FrameBuffer, ExtractedPeopleResults

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WebServerOutput(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, frame_rate, FPS=20, img_dim=(640, 480)):
        super(WebServerOutput, self).__init__(WebServerOutput.input_type, WebServerOutput.output_type)
        self.frame_rate = frame_rate
        self.frame_time = float(1)/self.frame_rate
        self.q = mp.Queue(maxsize=90)
        # self.q = None
        self.last_image = np.zeros((480, 720, 3))
        self.video_writer = cv2.VideoWriter('./output/video.avi', -1, FPS, (img_dim[0], img_dim[1]))
        self.video_reader = None

    def _data_ingest(self, data):

        imgs = data.get_drown_detection_images()

        for i in range(len(imgs)):
            self.video_writer.write(imgs[i])

        self.video_reader = cv2.VideoCapture('./output/video.avi')

        print('Compute Time: {}s'.format(time.time() - data.enqueue_time))

    def clear_buffer(self):
        self.q = mp.Queue(maxsize=1000)

    def get_frame(self):
        if self.video_reader is None:
            return self.last_image

        ret, frame = self.video_reader.read()

        return frame
