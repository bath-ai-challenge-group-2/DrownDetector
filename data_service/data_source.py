import time
import threading
import multiprocessing


from abc import abstractmethod


class DataSource:

    POLL_TIMER = 0.1

    def __init__(self, output_type):
        self.output_type = output_type
        self.output_buffer = multiprocessing.Queue(maxsize=10)
        self.thread = threading.Thread(target=self.run, args=())
        # self.thread = multiprocessing.Process(target=self.run, args=())
        self.running = False

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
        while self.running:
            self._data_ingest()

    @abstractmethod
    def _data_ingest(self):
        raise NotImplementedError()