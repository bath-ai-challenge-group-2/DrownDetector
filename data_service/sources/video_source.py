import cv2
import time
import torch
from data_utils.torch_utils import img_to_torch

from data_service import DataSource
from data_models import FrameBuffer


class VideoSource(DataSource):

    output_type = FrameBuffer

    def __init__(self, video_path, buffer_size=20):
        super(VideoSource, self).__init__(VideoSource.output_type)
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.video = cv2.VideoCapture(self.video_path)
        self.frameCount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.f_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.f_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.f_height, self.f_width, 3)

    def _data_ingest(self):
        frame_id = 0
        buffer = FrameBuffer(self.buffer_size, self.frame_size, frame_id)

        while True:
            if buffer.is_full():
                buffer.enqueue_time = time.time()
                self.enqueue(buffer)
                buffer = FrameBuffer(self.buffer_size, self.frame_size, frame_id)
                time.sleep(1)

            ret, img = self.video.read()

            if ret is False:
                self.video = cv2.VideoCapture(self.video_path)
                frame_id = 0
                continue

            buffer.add(img)
            frame_id += 1
