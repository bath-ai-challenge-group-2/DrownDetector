import time
import threading
import multiprocessing


from abc import abstractmethod


class DataOutput:

    POLL_TIMER = 0.1

    def __init__(self, input_type):
        self.input_type = input_type
        self.thread = threading.Thread(target=self.run, args=())
        self.running = False

    def __len__(self):
        return self.output_buffer.qsize()

    def enqueue(self, data):
        pass

    def register(self, output_type, input_source):
        assert output_type == self.input_type, "The output type of the previous data service and the input" \
                                                            "of this one must be the same, but they are not!"
        self.input_source = input_source

    def get_next(self):
        return self.output_buffer.get()

    def start(self):
        self.running = True
        self.thread.start()

    def run(self):
        while self.running:
            if self.input_source.empty():
                continue

            data = self.input_source.get()
            self._data_ingest(data)

    @abstractmethod
    def _data_ingest(self, data):
        raise NotImplementedError()
