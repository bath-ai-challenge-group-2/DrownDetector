class YoloResults:

    def __init__(self, frame_id, frames, results):
        self.frame_id = frame_id
        self.frames = frames
        self.results = results

    def get_next(self):
        for frame, result in zip(self.frames, self.results):
            yield frame, result

    def test_next(self):
        for i in range(self.results.n):
            yield self.results.imgs[i], self.results.pred[i], self.results.xyxy[i]
