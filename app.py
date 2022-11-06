# import gradio as gr
# from hydra import compose, initialize
# from omegaconf import OmegaConf
# from mt.transforms.albumentations_adapter import Adapter
# from mt.transforms.albumentations_utils import resize_and_pad
# import hydra
# import PIL
# from PIL import Image
# from mt.data.dataset import Dataset
# import albumentations as A
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from mt.utils.bbox_inverse_transform import inverse_transform_record
#
# import mt.models.yolov5
#
#
#
# def load_model(ckpt):
#     with initialize(config_path="configs"):
#         cfg = compose(config_name='train',
#                       overrides=['module=yolov5', 'module.backbone=small_p6', 'module.model.model.device.device=cpu'])
#
#     model = hydra.utils.instantiate(cfg.module.model)
#     model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
#     return model
#
#
# def predict_img(img, model, tfms):
#     ds = Dataset.from_images(images=[img], tfm=tfms)
#     dl = mt.models.yolov5.infer_dl(ds)
#     pred = mt.models.yolov5.predict_from_dl(model.model, dl, detection_threshold=0.05, keep_images=True)
#     pred = inverse_transform_record(pred[0])
#     return pred
#
#
# def draw_img(img, pred, confidence_threshold=0.1):
#     fig, ax = plt.subplots(figsize=(14, 10))
#     ax.imshow(img)
#     for bbox, score in zip(pred.detection.bboxes, pred.detection.scores):
#         if score < confidence_threshold:
#             break
#         xmin, ymin, xmax, ymax = bbox.xyxy
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
#         # ax.text(xmin, ymin, f"{score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
#     return fig2img(fig)
#
# def fig2img(fig):
#     """Convert a Matplotlib figure to a PIL Image and return it"""
#     import io
#     buf = io.BytesIO()
#     fig.savefig(buf)
#     buf.seek(0)
#     img = Image.open(buf)
#     return img
#
# if __name__ == '__main__':
#     # ckpt = '/home/kuntik/0.750.ckpt'
#     # tfms = Adapter([*resize_and_pad((1024, 896)), A.Normalize(std=(0.4, 0.4, 0.4), mean=(0.4, 0.4, 0.4))])
#     # img = Image.open('samples/dataset/images/1.png').convert('RGB')
#     # model = load_model(ckpt)
#     # pred = predict_img(img, model, tfms)
#     # final_img = draw_img(img, pred)
#     # final_img.save('final_img.png', format='png')
#     css = '''
#     h1#title {
#       text-align: center;
#     }
#     '''
#     demo = gr.Blocks(css=css)
#     title = """<h1 id="title">Bitewing caries detection</h1>"""
#
#     with demo:
#         gr.Markdown(title)
#         slider_input = gr.Slider(minimum=0.01, maximum=1, value=0.2, label='Confidence threshold')
#
#         with gr.Tabs():
#             with gr.TabItem('Image URL'):
#                 with gr.Row():
#                     url_input = gr.Textbox(lines=2, label='Enter valid image URL here..')
#                     img_output_from_url = gr.Image(shape=(650, 650))
#
#                     with gr.Row():
#                         example_url = gr.Dataset(components=[url_input], samples=[[str(url)] for url in urls])
#
#                     url_but = gr.Button('Detect')
#
#                 with gr.TabItem('Image Upload'):
#                     with gr.Row():
#                         img_input = gr.Image(type='pil')
#                         img_output_from_upload = gr.Image(shape=(650, 650))
#
#                     with gr.Row():
#                         example_images = gr.Dataset(components=[img_input],
#                                                     samples=[[path.as_posix()]
#                                                              for path in sorted(pathlib.Path('images').rglob('*.JPG'))])
#
#                     img_but = gr.Button('Detect')
#
#             url_but.click(detect_objects, inputs=[options, url_input, img_input, slider_input],
#                           outputs=img_output_from_url, queue=True)
#             img_but.click(detect_objects, inputs=[options, url_input, img_input, slider_input],
#                           outputs=img_output_from_upload, queue=True)
#             example_images.click(fn=set_example_image, inputs=[example_images], outputs=[img_input])
#             example_url.click(fn=set_example_url, inputs=[example_url], outputs=[url_input])
#
#     demo.launch(enable_queue=True)

import gradio as gr

def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
