class ExtractedPeopleResults:

    def __init__(self, enqueue_time, frame_id, frames):
        self.enqueue_time = enqueue_time
        self.frame_id = frame_id
        self.frames = frames

        #From Yolo Detector
        self.imgs = None
        self.preds = None
        self.xyxy = None

        #From Yolo PostPro
        self.pose_inpts = None

        #From Pose Estimation
        self.pose_preds = None

        #From tracking system
        self.tracked_frames = None

        self.joint_locations = None

        #From drown detection images
        self.final_frames = None


    def add_yolo_detection(self, results):
        self.imgs = results.imgs
        self.preds = results.pred
        self.xyxy = results.xyxy

    def get_yolo_results(self):
        assert self.imgs is not None and self.preds is not None and self.xyxy is not None, "Attempting to access yolo results" \
                                                                                           "before yolo has been run!"
        for i in range(len(self.imgs)):
            yield self.imgs[i], self.preds[i], self.xyxy[i]

    def add_yolo_post_pro(self, results):
        self.pose_inpts = results

    def get_pose_inpts(self):
        assert self.pose_inpts is not None, "Attempting to access yolo post process results, before results have been" \
                                            "processed!"

        return self.pose_inpts

    def add_pose_estimation_predictions(self, results):
        self.pose_preds = results

    def get_pose_preds(self):
        assert self.pose_preds is not None, "Attempting to access pose estimation results before AlphaPose has been " \
                                            "run!"

        return self.pose_preds

    def get_frame_data(self):
        assert self.pose_inpts is not None, "Attempting to access yolo post process results, before results have been" \
                                            "processed!"

        for i in range(len(self.imgs)):
            yield self.imgs[i], self.preds[i], self.xyxy[i]

    def add_tracked_frames(self, results):
        self.tracked_frames = results

    def get_tracked_frames(self):
        assert self.tracked_frames is not None, "Attempting to access tracked positions before results have been" \
                                            "processed!"

        for i in range(len(self.tracked_frames)):
            yield self.imgs[i], self.xyxy[i], self.tracked_frames[i]

    def add_drown_detection_images(self, results):
        self.final_frames = results

    def get_drown_detection_images(self):
        assert self.final_frames is not None, "Attempting to access pose estimation results before AlphaPose has been " \
                                            "run!"

        return self.final_frames