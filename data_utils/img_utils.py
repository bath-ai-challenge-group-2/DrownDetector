import cv2

import numpy as np


def padded_crop_image(img, box, padding=5):

    # width, heigth = box[2]-box[0], box[3]-box[1]
    new_image = np.zeros((img.shape[0]+padding*2, img.shape[1]+padding*2, img.shape[2]), dtype=np.uint8)
    new_image[padding:-padding, padding:-padding, :] = img
    box += padding

    return new_image[int(box[1])-padding:int(box[3])+padding, int(box[0])-padding:int(box[2])+padding, :]


def letterbox_scale(img, new_dim):
    """Scales an RGB Image to a desired dimension using
    a letter box"""

    height, width, _ = img.shape
    ratio = height/width
    desired_ratio = new_dim[0]/new_dim[1]

    if ratio >= desired_ratio:
        n_height = new_dim[0]
        n_width = int(n_height / ratio)
    else:
        n_width = new_dim[1]
        n_height = int(n_width * ratio)

    res = cv2.resize(img, (n_width, n_height), interpolation=cv2.INTER_LINEAR)

    new_img = np.zeros((new_dim[0], new_dim[1], 3), dtype=np.uint8)

    mids = np.floor(new_dim/2)
    new_offsets = np.floor(np.array([res.shape[0]/2, res.shape[1]/2]))

    new_img[
        int(mids[0]-new_offsets[0]): int(mids[0]-new_offsets[0] + res.shape[0]),
        int(mids[1]-new_offsets[1]): int(mids[1]-new_offsets[1] + res.shape[1]),
        :
    ] = res

    return new_img