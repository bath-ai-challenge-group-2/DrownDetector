import collections
import threading
import ffmpeg
import cv2

import numpy as np


class IPCameraReciever:

    def __init__(self, address='tcp://localhost:40000', buffer_len=30, img_dim=(1280, 1280, 3),
                 frame_callback=None, error_callback=None):
        self.sock = (
            ffmpeg
            .input(address, listen=1)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        self.connected = False
        self.img_dim = img_dim
        self.frame_buffer = np.zeros((buffer_len,) + img_dim, dtype=np.uint8)
        self.frame_counter = 0
        self.ptr = 0
        self.buffer_len = buffer_len
        self.frame_callback = frame_callback
        self.error_callback = error_callback
        self.update_lock = threading.Lock()

    def start(self):
        # self.run()
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def run(self):
        self.connected = True

        while True:
            in_bytes = self.sock.stdout.read(1280 * 720 * 3)
            if not in_bytes:
                self.close()
                if self.error_callback is not None: self.error_callback('Connection to client broken')
                return
            in_frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([720, 1280, 3])
            )
            frame = in_frame[..., ::-1]
            new_frame = np.zeros((1280, 1280, 3))
            new_frame[280:1000,:,:] = frame

            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(self.ptr)
            self.frame_buffer[self.ptr] = new_frame

            self.ptr += 1
            if self.ptr >= self.buffer_len:
                self.ptr = 0

            # self.frame_callback(self.frame_buffer[self.ptr])

    def get_current_frame(self):
        return self.ptr, self.frame_buffer[self.ptr].copy()

    def close(self):
        self.thread.join()
