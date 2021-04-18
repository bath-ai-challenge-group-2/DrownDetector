import time
import queue
# import threading
import torch.multiprocessing as mp

# ctx = mp.get_context('spawn')
#
# from torch.multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

from abc import abstractmethod


class DataService:

    POLL_TIMER = 0.1

    def __init__(self, input_type, output_type):
        self.input_type = input_type
        self.output_type = output_type
        self.output_buffer = mp.Queue(maxsize=30)
        # self.thread = threading.Thread(target=self.run, args=())
        self.thread = mp.Process(target=self.run, args=())
        self.running = False
        self.input_source = None

    def register(self, output_type, input_source):
        assert output_type == self.input_type, "The output type of the previous data service and the input" \
                                                            "of this one must be the same, but they are not!"
        self.input_source = input_source

    def __len__(self):
        return self.output_buffer.qsize()

    def enqueue(self, data):
        assert isinstance(data, self.output_type)
        self.output_buffer.put(data)

    def get_next(self):
        return self.output_buffer.get()

    def start(self):
        self.running = True
        # self.thread.daemon = True
        self.thread.start()

    def run(self):
        # print('Running')
        while self.running:
            # print(self.input_source.qsize())
            if self.input_source.empty():
                # time.sleep(DataService.POLL_TIMER)
                continue

            data = self.input_source.get()
            # start_time = time.time()
            self._data_ingest(data)
            # process_time = time.time() - start_time
            # print('YOLO Process Time: {}s'.format(process_time))

    @abstractmethod
    def _data_ingest(self, data):
        raise NotImplementedError()