# import gradio as gr
import albumentations as A
import hydra
import matplotlib.pyplot as plt
import torch
from PIL import Image
from hydra import compose, initialize
from torch import nn
from pathlib import Path

import mt.models.yolov5
from mt.data.dataset import Dataset
from mt.transforms.albumentations_adapter import Adapter
from mt.transforms.albumentations_utils import resize_and_pad
from mt.utils.bbox_inverse_transform import inverse_transform_record
from mt.utils.visualization import fig2img


class Inference:
    def __init__(self):
        self.model: nn.Module
        self.img: Image = None
        self.tfms = Adapter([*resize_and_pad((1024, 896)), A.Normalize(mean=0.367, std=0.277)])
        self.confidence_threshold: float = 0.5

    def set_confidence(self, confidence_threshold, redraw=False):
        self.confidence_threshold = confidence_threshold
        if redraw:
            return self.draw_img()

    def model_from_cfg(self, cfg, ckpt=None):
        self.model = hydra.utils.instantiate(cfg)
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])

    def load_yolov5(self, ckpt=None, backbone='small_p6'):
        with initialize(config_path="configs", version_base='1.1'):
            cfg = compose(config_name='train',
                          overrides=['module=yolov5', f'module.backbone={backbone}',
                                     'module.model.model.device.device=cpu', 'module.model.model.backbone.pretrained=0'])

        self.model = hydra.utils.instantiate(cfg.module.model)
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])

    def predict(self, img: Image, nms_iou_threshold=0.4, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img = img
        ds = Dataset.from_images(images=[img], tfm=self.tfms)
        dl = mt.models.yolov5.infer_dl(ds)
        pred = mt.models.yolov5.predict_from_dl(self.model.model, dl, detection_threshold=0.01, keep_images=True, nms_iou_threshold=nms_iou_threshold)
        self.pred = inverse_transform_record(pred[0])

    def draw_img(self):
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(self.img)
        for bbox, score in zip(self.pred.detection.bboxes, self.pred.detection.scores):
            if score < self.confidence_threshold:
                break
            xmin, ymin, xmax, ymax = bbox.xyxy
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
        ax.axis('off')
        return fig2img(fig)



if __name__ == '__main__':
    ckpt = '/home/kuntik/dev/BitewingCariesDetection/yolo_small.ckpt'
    tfms = Adapter([*resize_and_pad((1024, 896)), A.Normalize(std=(0.4, 0.4, 0.4), mean=(0.4, 0.4, 0.4))])
    img = Image.open('../samples/dataset/images/1.png').convert('RGB')
    inference = Inference()
    inference.load_yolov5(ckpt=ckpt)
    inference.predict(img)
    final_img = inference.draw_img()
    final_img.save('final_img.png', format='png')
