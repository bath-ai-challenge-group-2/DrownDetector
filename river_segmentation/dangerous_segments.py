import cv2


class DangerousSegments:

    def __init__(self, segments):
        self.segments = segments

    def check_in_segment(self, pts):
        for pt in pts:
            for seg in self.segments:
                result = cv2.pointPolygonTest(seg, (pt[0], pt[1]), False)
                if result > 0:
                    return True
        return False

    def get_segments(self):
        return self.segments
