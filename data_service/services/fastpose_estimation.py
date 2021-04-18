# import time
# import torch
#
# import numpy as np
#
# from data_utils.torch_utils import img_to_torch
#
# from data_service.data_service import DataService
# # from data_service.models.small_fast_pose import SmallFastPose
# from data_models import FrameBuffer, ExtractedPeopleResults
#
# # from alphapose-older.utils.config import update_config
# # from alphapose-older.models import builder
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# class AlphaFastPoseEstimation(DataService):
#
#     input_type = ExtractedPeopleResults
#     output_type = ExtractedPeopleResults
#
#     def __init__(self, model_path='./model_weights/sppe/fast_res50_256x192.pth'):
#         super(AlphaFastPoseEstimation, self).__init__(AlphaFastPoseEstimation.input_type, AlphaFastPoseEstimation.output_type)
#         self.pose_model = SmallFastPose().to(device)
#         self.pose_model.load_state_dict(torch.load(model_path))
#         self.pose_model.eval()
#
#     def _data_ingest(self, data):
#         inpts = data.get_pose_inpts()
#         inpts = np.concatenate(inpts)
#         inpts = torch.from_numpy(inpts).float().cuda()
#         inpts /= 255
#         # inpts.to(device)
#         with torch.no_grad():
#             results = self.pose_model(inpts)
#             results.cpu().detach().numpy()
#
#         data.add_pose_estimation_predictions(results)
#         self.enqueue(data)
#         print('Compute Time: {}s'.format(time.time() - data.enqueue_time))
#
#     # def _data_ingest(self, data):
#     #     inpts = data.get_pose_inpts()
#     #     # inpts = np.concatenate(inpts)
#     #
#     #     res = []
#     #
#     #     for inpt in inpts:
#     #         inp = torch.from_numpy(inpt).float().cuda()
#     #         inp /= 255
#     #         # inpts.to(device)
#     #         with torch.no_grad():
#     #             results = self.pose_model(inp)
#     #             results.cpu().detach().numpy()
#     #             res.append(results)
#     #
#     #     data.add_pose_estimation_predictions(results)
#     #     self.enqueue(data)
#     #     print('Compute Time: {}s'.format(time.time() - data.enqueue_time))
