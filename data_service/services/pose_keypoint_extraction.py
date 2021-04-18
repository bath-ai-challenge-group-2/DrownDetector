import time
import torch

import numpy as np

from data_utils.torch_utils import img_to_torch

from data_service.data_service import DataService
from data_models import FrameBuffer, ExtractedPeopleResults

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PoseKeypointDetection(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, input_dim=(1280, 720), output_dim=(1280, 720)):
        super(PoseKeypointDetection, self).__init__(PoseKeypointDetection.input_type, PoseKeypointDetection.output_type)

    def _data_ingest(self, data):
        pass