import time
import numpy as np


class FrameBuffer:

    def __init__(self, buffer_length, frame_size, start_frame_id):
        self.enqueue_time = time.time()
        self.frame_size = frame_size
        self.buffer_length = buffer_length
        self.start_frame_id = start_frame_id
        self.ptr = 0
        self.buffer = [None] * self.buffer_length

    def is_full(self):
        return self.ptr >= self.buffer_length

    def add(self, frame):
        assert self.frame_size == frame.shape, "Frame is incorrect dimensions for the current buffer"
        self.buffer[self.ptr] = frame
        self.ptr += 1

    def get(self):
        return self.buffer
