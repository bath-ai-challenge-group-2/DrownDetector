import torch


def img_to_torch(img):
    return torch.from_numpy(img).float()