import numpy as np


class SimpleFIFOBuffer:

    def __init__(self, max_len=10):
        self.buffer = np.zeros((max_len, 2))
        self.buffer[:,:] = np.nan
        self.max_len = max_len
        self.ptr = 0

    def put(self, item):
        self.buffer[self.ptr] = item
        self.ptr += 1
        if self.ptr > self.max_len-1:
            self.ptr = 0

    def peek_buffer(self):
        if self.ptr > 0:
            new_arr = np.concatenate((self.buffer[self.ptr:], self.buffer[:self.ptr]))
            return new_arr
        else:
            return self.buffer.copy()
