from __future__ import annotations

__all__ = ["bboxes_width_height_histogram", "num_caries_histogram", "create_heatmap_from_pvalues", "pairwise_averaged_plot", "pairwise_plot", "pr_curve"]

from typing import Union
import seaborn as sns
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from mt.evaluation.comparison.pairwise_comparison import generate_data, pairwise_centroids_plot_data_per_seniority
from mt.evaluation.prediction_evaluation import PredictionEval
from mt.utils.conver_to_coco import to_coco

from mt.utils.visualization import set_fig_size
def bboxes_width_height_histogram(annotations: str| Path, max_size:int=150, n_bins:int=50, fig_size:int = 407) -> plt.Figure:
    bins = np.linspace(0, max_size, n_bins)
    width = []
    height = []

    with open(annotations) as f:
        data = json.load(f)

    for ann in data['annotations']:
        _,_,w,h = ann['bbox']

        width.append(w)
        height.append(h)

    fig, ax = plt.subplots(1,1, figsize=set_fig_size(fig_size))
    ax.hist(width, bins, alpha=0.8, label='width',histtype='step',linewidth=3)
    ax.hist(height, bins, alpha=0.5, label='height',histtype='step',linewidth=1)
    ax.set_xlabel('size [pixels]')
    ax.set_ylabel('number of bounding boxes')
    ax.legend()
    ax.set_xlim([0,max_size])
    fig.savefig('images/dataset_histogram2.pdf')
    return fig


def num_caries_histogram(annotations: str| Path, fig_size:int = 407) -> plt.Figure:

    with open(annotations) as f:
        data = json.load(f)
    images = [0 for i in range(len(data['images']))]
    for ann in data['annotations']:
        images[ann['image_id']] += 1

    fig, ax = plt.subplots(1, 1, figsize=set_fig_size(fig_size))
    bins = np.linspace(0, 10, 11)
    plt.hist(images, bins,histtype='step',align='left',linewidth=2)
    ax.set_xlabel('number of dental caries per image')
    ax.set_ylabel('number of images')
    ax.set_xlim([0, 10])
    ax.set_xticks(list(range(0,10)))
    fig.savefig('images/caries_histogram2.pdf')
    return fig

def create_heatmap_from_pvalues(p_values: np.ndarray, names:list[str], fig_size:int=407) -> plt.Figure:
    fig, ax = plt.subplots(figsize=set_fig_size(fig_size))
    import matplotlib.cm as mpl_cm
    cmap_cont = sns.diverging_palette(20, 120, as_cmap=True,)
    # cpal = 'PuOr_r'
    # cmap_cont = mpl_cm.get_cmap(cpal)
    cmap_cont.set_over(color='green')
    cmap_cont.set_under(color='red')
    # cmap.set_over()
    sns.heatmap(p_values, cmap=cmap_cont, xticklabels=names, yticklabels=names, ax=ax, vmin=-0.95, vmax=0.95, cbar_kws={'extend':'both', 'ticks' : [-0.95,-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.95]})
    return fig


def pairwise_averaged_plot(annotations_path, model_data_path, average_over: str = "experts", fig_size=407):
    with open(model_data_path, 'r') as f:
        model = json.load(f)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    evaluate_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    experts = [1, 2, 3, 4, 5]
    novices = [6, 7, 8]
    names_dict = {1: "$E_0$", 2: "$E_1$", 3: "$E_2$", 4: "$E_3$", 5: "$E_4$", 6: "$N_1$", 7: "$N_2$", 8: "$N_3$",
                  0: "$M$"}
    data_dict = {0: model}

    names, ious, errors = generate_data(annotations_path, ids, data_dict, evaluate_ids, iou_threshold=0.0,
                                        per_img=False)
    e_merged, i_merged, n_merged, c_merged = pairwise_centroids_plot_data_per_seniority(average_over, errors, ious,
                                                                                        names, ids, experts, novices,
                                                                                        names_dict)
    e_merged, i_merged, n_merged, c_merged = np.array(e_merged), np.array(i_merged), np.array(n_merged), np.array(
        c_merged)

    fig, ax = plt.subplots(figsize=set_fig_size(fig_size))
    # colors = ListedColormap(['b','r','g', 'm'])
    color = ['b', 'r', 'g', 'm']
    marker_dict = {0: 'o', 1: "v", 2: "^", 3: "s"}

    for i in range(4):
        em = e_merged[c_merged == i]
        im = i_merged[c_merged == i]
        nm = n_merged[c_merged == i]
        ax.scatter(x=em, y=im, c=color[i], s=30, marker=marker_dict[i])

        for i, txt in enumerate(nm):
            # if txt == '$M_1$':
            #     ax.annotate(txt, (e_merged[i]+1, i_merged[i]-0.01))
            # else:
            ax.annotate(txt, (em[i] + 1, im[i] + 0.001))

    # ax.legend(handles=scatter.legend_elements()[0], labels=['All', 'Novices', 'Experts'], title='Compared with:', loc='lower right')
    ax.set_ylabel('Average IOU')
    ax.set_xlabel('Total number of errors')
    return fig



def pairwise_plot(annotations_path, model_data_path, fig_size: int = 407) -> plt.Figure:
    with open(model_data_path, 'r') as f:
        model = json.load(f)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    evaluate_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    names_dict = {1: "$E_0$", 2: "$E_1$", 3: "$E_2$", 4: "$E_3$", 5: "$E_4$", 6: "$N_1$", 7: "$N_2$", 8: "$N_3$",
                  0: "$M$"}
    data_dict = {0: model}

    names, ious, errors = generate_data(annotations_path, ids, data_dict, evaluate_ids, iou_threshold=0.0,
                                        per_img=False)

    names_text = [names_dict[eval(n)[0]] + '/' + names_dict[eval(n)[1]] for n in names]
    names_text, ious, errors = np.array(names_text), np.array(ious), np.array(errors)
    colors = []
    # offsets for point labels to avoid overlaps, in units of error and IoU
    hints={"$E_1$/$N_2$": (-2,0.015),
           "$E_0$/$E_1$": (-3,-0.02),
           "$N_1$/$N_2$": (-2,-0.02),
           "$E_2$/$N_1$": (-4,-0.02),
           "$E_4$/$N_2$": (-3,-0.02),
           "$M$/$N_2$": (-4,0.01),
           "$M$/$N_1$": (2,-0.005),
           "$E_0$/$N_2$": (-2,-0.02),
           "$E_0$/$E_3$": (-2,-0.02),
}
    

    for n in names_text:
        if 'M$/$E_0' in n:
            colors.append(0)
        elif 'E' in n and 'M' in n:
            colors.append(1)
        elif 'M' in n:
            colors.append(2)
        else:
            colors.append(3)

    colors = np.array(colors)
    cmap = ['b', 'g', 'm', '0.25']
    markers = ['v', '^', 's', 'o']
    labels = ['Annotator', 'Expert', 'Novice', '']

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx in range(4):
        em = errors[colors == idx]
        im = ious[colors == idx]
        nm = names_text[colors == idx]
        ax.scatter(x=em, y=im, c=cmap[idx], marker=markers[idx], label=labels[idx])

        for e, i, n in zip(em, im, nm):
            #print(f"annotating '{n}'")
            try:
               offsets=hints[n]
               #print(f"    hint for offset found: {offsets}")
            except KeyError:
               offsets=(2,0)
            ax.annotate(n, xy=(e + offsets[0], i+offsets[1]))
    ax.set_xlabel('Number of errors')
    ax.set_ylabel('Average IOU')
    ax.legend(title='Automatic method compared with:', loc='lower right')
    return fig

def pr_curve(annotations_path: str|Path, data_names:list[str], data_paths: list[str|Path], fig_size:int = 407) -> plt.Figure:
    plot_data = {}
    with open(annotations_path, 'r') as f:
        ann_data = json.load(f)

    for name, path in zip(data_names, data_paths):
        with open(path, 'r') as f:
            data = json.load(f)

        preds_coco, names = to_coco(data, ann_data)
        eval = PredictionEval()
        eval.load_data_coco_files(annotations_path, preds_coco, names)
        plot_data[name] = eval.precision_recall_score()

    line_styles = ['-', ':', '--', '-.']
    fig, ax = plt.subplots(figsize=set_fig_size(fig_size))
    for i, (keys, values) in enumerate(plot_data.items()):
        ax.plot(values[0], values[1], label=keys, linestyle=line_styles[i % len(line_styles)])
        ax.legend()
    return fig

