from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import gradio as gr
from copy import deepcopy
from pathlib import Path
import torch

from mt.inference import Inference

ckpt = 'yolo_small.ckpt'
inference = Inference()
inference.load_yolov5(ckpt=ckpt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(img):
    inference.predict(img, nms_iou_threshold=0.3)
    d = inference.draw_img()
    d.save('foo.png', format='png')
    return d

def change_model(model_name):
    if model_name == 'Fast':
        inference.load_yolov5(ckpt='yolo_small.ckpt', backbone='small_p6')
    if model_name == 'Normal':
        inference.load_yolov5(ckpt='yolo_medium.ckpt', backbone='medium_p6')
    if model_name == 'Slow':
        inference.load_yolov5(ckpt=ckpt, backbone='large_p6')
    inference.model.to(device)


def change_confidence(confidence):
    return inference.set_confidence(confidence, redraw=True)


with gr.Blocks() as app:
    # with gr.Row():
    inputs = gr.Image(type='pil', label='Input bitewing image')
    outputs = gr.Image(type='pil', label='Processed bitewing image')
    examples = gr.Examples(
        # [['samples/1.png'],['samples/2.png'],['samples/3.png'],['samples/4.png'],
        #  ['samples/5.png'], ['samples/6.png'], ['samples/7.png'], ['samples/8.png']],

        [['35.png'], ['2.png'], ['3.png'], ['4.png'],
         ['17.png'], ['27.png'], ['30.png'], ['1.png']],
        # 'samples/9.png','samples/10.png', 'samples/11.png', ['samples/12.png']],
        inputs=inputs, outputs=outputs, label='Gallery of examples')
    inputs.change(predict, inputs, outputs)
    slider_input = gr.Slider(minimum=0.05, maximum=1, value=0.1, label='Prediction Threshold')
    slider_input.change(change_confidence, inputs=slider_input, outputs=outputs)
    dropdown = gr.Dropdown(['Fast', 'Normal'], value='Normal', label='Choose a model type')
    dropdown.change(change_model, inputs=dropdown, outputs=outputs)

    if __name__ == '__main__':
        app.launch()