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

    # def __init__(self, address='tcp://192.168.0.67:40000', buffer_len=30, img_dim=(1080, 1920, 3),
    #              frame_callback=None, error_callback=None):
    #     self.sock = (
    #         ffmpeg
    #             .input(address)
    #             .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    #             .run_async(pipe_stdout=True)
    #     )
    #     self.connected = False
    #     self.img_dim = img_dim
    #     self.frame_buffer = np.zeros((buffer_len,) + img_dim, dtype=np.uint8)
    #     self.frame_counter = 0
    #     self.ptr = 0
    #     self.buffer_len = buffer_len
    #     self.frame_callback = frame_callback
    #     self.error_callback = error_callback
    #     self.update_lock = threading.Lock()

    def __init__(self, address='tcp://192.168.0.67:3333', buffer_size=20, img_dim=(1080, 1920, 3)):
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

        # while frame_id < self.frameCount:
        #     if buffer.is_full():
        #         print('Saving Buffer')
        #         self.enqueue(buffer)
        #         buffer = FrameBuffer(self.buffer_size, self.frame_size, frame_id)
        #
        #     ret, img = self.video.read()
        #
        #     if ret is False:
        #         break
        #
        #     buffer.add(img)
        #     frame_id += 1

        while True:
            if buffer.is_full():
                # print('Saving Buffer')
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

        # self.enqueue(buffer)