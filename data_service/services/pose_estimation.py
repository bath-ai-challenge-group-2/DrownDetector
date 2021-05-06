import time
import torch

import numpy as np

from data_utils.torch_utils import img_to_torch

from data_service.data_service import DataService
from data_models import FrameBuffer, ExtractedPeopleResults

from alphapose.utils.config import update_config
from alphapose.models import builder

# import sys
# sys.path.insert(0, './yolov5_master')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AlphaPoseEstimation(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, model_path='ultralytics/yolov5', model_type='yolov5s'):
        super(AlphaPoseEstimation, self).__init__(AlphaPoseEstimation.input_type, AlphaPoseEstimation.output_type)
        cfg = update_config('model_weights/sppe/model.cfg')
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load('./model_weights/sppe/fast_res50_256x192.pth'))
        self.pose_model.eval()
        self.pose_model.cuda()

    def _data_ingest(self, data):
        inpts = data.get_pose_inpts()
        inpts = np.concatenate(inpts)
        inpts = torch.from_numpy(inpts).float().cuda()
        inpts /= 255
        with torch.no_grad():
            results = self.pose_model(inpts)
            results.cpu().detach().numpy()

        data.add_pose_estimation_predictions(results)
        self.enqueue(data)
        print('Compute Time: {}s'.format(time.time() - data.enqueue_time))

