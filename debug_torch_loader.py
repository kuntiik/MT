import torch
from src.models.yolov5 import *
torch.load('/home/kuntik/Desktop/foo.ckpt', map_location=torch.device('cpu'))
