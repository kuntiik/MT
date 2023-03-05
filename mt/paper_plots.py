from typing import Union
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

from mt.utils.visualization import set_fig_size
def bboxes_width_height_histogram(annotations: Union[str, Path], max_size:int=150, n_bins:int=50, fig_size:int = 407):
    bins = np.linspace(0, max_size, n_bins)
    width = []
    height = []

    with open('/datagrid/personal/kuntluka/dental_rtg/caries6.json') as f:
        data = json.load(f)

    for ann in data['annotations']:
        _,_,w,h = ann['bbox']

        width.append(w)
        height.append(h)

    fig, ax = plt.subplots(1,1, figsize=set_fig_size(407))
    ax.hist(width, bins, alpha=0.8, label='width')
    ax.hist(height, bins, alpha=0.5, label='height')
    ax.set_xlabel('size [pixels]')
    ax.set_ylabel('amount of bounding boxes')
    ax.legend()
    ax.set_xlim([0,max_size])
    return fig


