import random, os
import cv2, torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from EfficientViT.classification.model.build import EfficientViT_M5


class EfficientViT(nn.Module):
    def __init__(self, in_size, predict_change=False):
        super(EfficientViT, self).__init__()
        self.predict_change = predict_change
        self.evit = EfficientViT_M5(pretrained='efficientvit_m5')
        # remove the classification head
        self.evit = torch.nn.Sequential(*list(self.evit.children())[:-1])

    def forward(self, x):
        out = self.evit(x)
        decoded = self.decoder.forward(out)
        return decoded

def process_inputs(images):
    # flip color channels
    tmp = images[:, 0, :, :].clone()
    images[:, 0, :, :] = images[:, 2, :, :]
    images[:, 2, :, :] = tmp
    return images


if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)

    batch_size = 16 # set to anything
    device = "cuda:0" # set to anything

    class GSViT(nn.Module):
        def __init__(self):
            super().__init__()
            gsvit = EfficientViT(in_size=batch_size)
            gsvit.load_state_dict(torch.load("GSViT.pkl"))
            self.gsvit = gsvit.gsvit.to(device)
            
        def forward(self, x):
            x = process_inputs(x) # flip color channels
            return self.gsvit(x)

    gsvit = GSViT()
    
    # write your training here
    # you can run gsvit in train or eval mode
    # e.g. gsvit.train(), gsvit.eval()
