import cv2
import time
import torch

import numpy as np

from data_utils.torch_utils import img_to_torch

from data_service.data_service import DataService
from data_models import FrameBuffer, ExtractedPeopleResults
from river_segmentation.segmenter import SegmentationMasks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RiverDetection(DataService):

    input_type = FrameBuffer
    output_type = ExtractedPeopleResults

    def __init__(self, path):
        super(RiverDetection, self).__init__(RiverDetection.input_type, RiverDetection.output_type)
        self.model = SegmentationMasks(path)

    def _data_ingest(self, data):
        for frame in range(data.get()):
            mask = self.model.mask(frame)

            cv2.imshow('random name', mask)
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
