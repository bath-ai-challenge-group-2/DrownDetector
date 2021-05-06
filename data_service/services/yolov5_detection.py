import torch

import numpy as np

from data_utils.torch_utils import img_to_torch

from data_service.data_service import DataService
from data_models import FrameBuffer, ExtractedPeopleResults

import sys
sys.path.insert(0, './yolov5_master')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class YoloV5Detection(DataService):

    input_type = FrameBuffer
    output_type = ExtractedPeopleResults

    def __init__(self, model_path='ultralytics/yolov5', model_type='yolov5s'):
        super(YoloV5Detection, self).__init__(YoloV5Detection.input_type, YoloV5Detection.output_type)
        self.model = torch.hub.load(model_path, model_type)
        self.model.to(device)

    def _data_ingest(self, data):
        proc_data = data.get()
        with torch.no_grad():
            results = self.model(proc_data, size=640)
        preds = []
        xyxys = []
        for i in range(len(results.pred)):
            p_reds = results.pred[i].cpu().detach().numpy()
            x_yxy = results.xyxy[i].cpu().detach().numpy()

            pred = []
            xyxy = []
            for j in range(len(results.pred[i])):
                pred.append(p_reds[j])
                xyxy.append(x_yxy[j])
            preds.append(np.array(pred))
            xyxys.append(np.array(xyxy))
        results.pred = preds
        results.xyxy = xyxys
        results.xywh = None
        results.xywhn = None
        results.xyxyn = None
        results.s = None

        yolo_results = ExtractedPeopleResults(data.enqueue_time, data.start_frame_id, data.get())
        yolo_results.add_yolo_detection(results)
        self.enqueue(yolo_results)
