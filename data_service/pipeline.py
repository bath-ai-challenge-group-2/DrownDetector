from .data_service import DataService
from .data_source import DataSource
from .data_output import DataOutput


class DataPipeline:

    def __init__(self):
        self.pipeline = []

    def add_service(self, service):
        assert isinstance(service, DataService) or isinstance(service, DataSource) or isinstance(service, DataOutput), "All Pipeline services must extend" \
                                                                                    "the service base classes!"

        if len(self.pipeline) < 1:
            self.pipeline.append(service)
            return

        if service.input_type is self.pipeline[-1].output_type:
            service.register(self.pipeline[-1].output_type, self.pipeline[-1].output_buffer)
            self.pipeline.append(service)

    def start(self):
        for service in self.pipeline:
            service.start()
