import cv2
import time
import torch
import ffmpeg

import numpy as np

from data_utils.torch_utils import img_to_torch

from data_service import DataSource
from data_models import FrameBuffer


class IPCamera(DataSource):

    output_type = FrameBuffer

    def __init__(self, address='tcp://localhost:40003', buffer_size=20, img_dim=(1080, 1920, 3)):
        super(IPCamera, self).__init__(IPCamera.output_type)
        self.sock = (
            ffmpeg
            .input(address)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        self.buffer_size = buffer_size
        self.frame_size = img_dim

    def _data_ingest(self):
        frame_id = 0
        buffer = FrameBuffer(self.buffer_size, self.frame_size, frame_id)

        while True:
            if buffer.is_full():
                buffer.enqueue_time = time.time()
                self.enqueue(buffer)
                buffer = FrameBuffer(self.buffer_size, self.frame_size, frame_id)

            in_bytes = self.sock.stdout.read(self.frame_size[0] * self.frame_size[1] * self.frame_size[2])
            if not in_bytes:
                return
            in_frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape(np.array(self.frame_size))
            )
            frame = in_frame[..., ::-1]

            buffer.add(frame)
            frame_id += 1
