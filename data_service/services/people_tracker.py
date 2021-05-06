import cv2
import copy
import time
import torch

import numpy as np

from scipy.spatial.distance import cdist

from data_service.data_service import DataService
from data_utils.datatypes import SimpleFIFOBuffer
from data_models import FrameBuffer, ExtractedPeopleResults

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colours = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (0, 0, 0),
    (255, 255, 255),
    (0, 255, 255),
    (0, 0, 0),
    (255, 255, 255),
]


class PeopleTracker(DataService):

    input_type = ExtractedPeopleResults
    output_type = ExtractedPeopleResults

    def __init__(self, distance_threshold=20):
        super(PeopleTracker, self).__init__(PeopleTracker.input_type, PeopleTracker.output_type)
        self.distance_threshold = distance_threshold
        self.people = []
        self.img_counter = 0

    def _data_ingest(self, data):
        tracked_frames = []
        for img, preds, xyxy in data.get_frame_data():

            if(len(preds)) < 1:
                tracked_frames.append([])
                continue

            confs = np.where(preds[:, 4] > 0.75)

            _xyxy = preds[confs]

            x_centre = (_xyxy[:, 0] + _xyxy[:, 2])/2
            y_centre = _xyxy[:, 3]
            centre = np.vstack((x_centre, y_centre)).T

            if len(self.people) > 0:
                last_pos = np.array([pple.current_pos for pple in self.people])
                seen = np.array([False for _ in range(len(self.people))])
                dists = cdist(last_pos, centre, 'euclidean')

                for i in range(len(self.people)):
                    try:
                        if len(dists) < 1:
                            continue
                        idx = np.unravel_index(dists[seen == False].argmin(), dists.shape)
                        dist = dists[idx]
                        if dist < self.distance_threshold:
                            self.people[idx[0]].add_new_position(centre[idx[1]])
                            seen[idx[0]] = True
                            dists[:, idx[1]] = np.inf
                    except:
                        continue
                self.people = [self.people[j] for j in range(len(self.people)) if seen[j]]

                dsts = np.where(dists < np.inf)

                if dsts is None:
                    continue

                for i in range(len(dsts[1])):
                    person = Person(centre[dsts[1][i]])
                    self.people.append(person)


            else:
                for i in range(centre.shape[0]):
                    person = Person(centre[i])
                    self.people.append(person)

            self.img_counter += 1
            tracked_frames.append(copy.deepcopy(self.people))
        data.add_tracked_frames(tracked_frames)
        self.enqueue(data)

class Person:

    Current_ID = 0

    def __init__(self, pos):
        self.id = Person.Current_ID
        self.pos_history = SimpleFIFOBuffer(max_len=50)
        self.add_new_position(pos)
        Person.Current_ID += 1

    def add_new_position(self, pos):
        self.current_pos = pos
        self.pos_history.put(pos)

    def get_pos_history(self):
        return self.pos_history.peek_buffer()
