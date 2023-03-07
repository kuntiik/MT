import random
from .core import Comparison
from itertools import combinations
from typing import List
import numpy as np
from pathlib import Path
import json
from scipy.stats import wilcoxon


def generate_data(coco_data_path: str, ids: List[int], data_json_dict: dict, evaluate_ids=None, per_img=False,
                  iou_threshold: float = 0.0):
    names, errors, ious = [], [], []
    if evaluate_ids is None: evaluate_ids = ids
    if type(coco_data_path) == Path:
        coco_data_path = str(coco_data_path)

    comparison = Comparison()
    comparison.parse_coco_data(coco_data_path)
    for c in combinations(ids, 2):
        id1, id2 = c
        if id1 not in data_json_dict.keys():
            comparison.load_coco_data(id1, True)
        else:
            comparison.load_json_data(data_json_dict[id1], True)

        if id2 in evaluate_ids:
            if id2 not in data_json_dict.keys():
                comparison.load_coco_data(id2, False)
            else:
                comparison.load_json_data(data_json_dict[id2], False)
        if per_img:
            iou, e = comparison.pairwise_evaluate_per_img(iou_threshold=iou_threshold)
        else:
            iou, e = comparison.pairwise_evaluate(iou_threshold=iou_threshold)
        # e = fp + fn
        names.append(str(c))
        ious.append(iou)
        errors.append(e)
    return names, ious, errors


def generate_pairwise_table(names, ious, errors, ids, id_dict=None):
    results = np.zeros((len(ids), len(ids)))
    id_dict = {id: index for index, id in enumerate(ids)} if id_dict is None else id_dict
    avg_iou = [0 for _ in ids]
    avg_e = [0 for _ in ids]
    for e, i, n in zip(errors, ious, names):
        i1, i2 = eval(n)
        avg_iou[id_dict[i1]] += i
        avg_iou[id_dict[i2]] += i
        avg_e[id_dict[i1]] += e
        avg_e[id_dict[i2]] += e
        if id_dict[i1] < id_dict[i2]:
            results[id_dict[i1], id_dict[i2]] = round(i, 2)
            results[id_dict[i2], id_dict[i1]] = e
        else:
            results[id_dict[i2], id_dict[i1]] = round(i, 2)
            results[id_dict[i1], id_dict[i2]] = e
    for i in range(len(ids)):
        avg_iou[i] /= (len(ids) - 1)
        avg_e[i] /= (len(ids) - 1)
    results = np.hstack([results, np.expand_dims(np.round(avg_iou, 2), 1)])
    results = np.vstack([results, np.hstack([np.asarray(avg_e), [0]])])
    # results = np.hstack([results, np.expand_dims(np.asarray(avg_e), 1)])
    # results = np.vstack([results, np.hstack([np.round(avg_iou,2), [0]])])
    return results


def generate_prob_table(names, ious, errors, ids):
    results = np.zeros((len(ids), len(ids)))
    id_dict = {id: index for index, id in enumerate(ids)}
    for e, i, n in zip(errors, ious, names):
        i1, i2 = eval(n)

        results[id_dict[i1], id_dict[i2]] = round(i, 2)
        results[id_dict[i2], id_dict[i1]] = e


def bootstrap_realization(iou1, iou2, e1, e2):
    num_e = 0
    num_i = 0
    for i in range(1000):
        sample_iou1 = random.choices(iou1, k=100)
        sample_iou2 = random.choices(iou2, k=100)
        sample_e1 = random.choices(e1, k=100)
        sample_e2 = random.choices(e2, k=100)
        if sample_iou1 > sample_iou2: num_i += 1
        if sample_e1 < sample_e2: num_e += 1

    return num_i / 1000.0, num_e / 1000.0


def pairwise_centroids_plot_data_per_seniority(seniority, errors, ious, names, ids, experts, novices, names_dict):
    centroids_e_all = {id: [] for id in ids}
    centroids_i_all = {id: [] for id in ids}
    centroids_e_expert = {id: [] for id in ids}
    centroids_i_expert = {id: [] for id in ids}
    centroids_e_novice = {id: [] for id in ids}
    centroids_i_novice = {id: [] for id in ids}

    for e, i, n in zip(errors, ious, names):
        i1, i2 = eval(n)
        if i1 in experts or i1 in novices:
            centroids_e_all[i2].append(e)
            centroids_i_all[i2].append(i)
            if i1 in experts:
                centroids_e_expert[i2].append(e)
                centroids_i_expert[i2].append(i)
            if i1 in novices:
                centroids_e_novice[i2].append(e)
                centroids_i_novice[i2].append(i)

        if i2 in experts or i2 in novices:
            centroids_e_all[i1].append(e)
            centroids_i_all[i1].append(i)
            if i2 in experts:
                centroids_e_expert[i1].append(e)
                centroids_i_expert[i1].append(i)
            if i2 in novices:
                centroids_e_novice[i1].append(e)
                centroids_i_novice[i1].append(i)

    for key, value in centroids_e_all.items():
        centroids_e_all[key] = sum(value) / len(value)
    for key, value in centroids_i_all.items():
        centroids_i_all[key] = sum(value) / len(value)
    for key, value in centroids_e_novice.items():
        centroids_e_novice[key] = sum(value) / len(value)
    for key, value in centroids_i_novice.items():
        centroids_i_novice[key] = sum(value) / len(value)
    for key, value in centroids_e_expert.items():
        centroids_e_expert[key] = sum(value) / len(value)
    for key, value in centroids_i_expert.items():
        centroids_i_expert[key] = sum(value) / len(value)

    names_centroids_all = [names_dict[i] for i in ids]
    names_centroids_novice = [names_dict[i] for i in ids]
    names_centroids_expert = [names_dict[i] for i in ids]

    centroids_e_all = [v for v in centroids_e_all.values()]
    centroids_i_all = [v for v in centroids_i_all.values()]
    centroids_e_novice = [v for v in centroids_e_novice.values()]
    centroids_i_novice = [v for v in centroids_i_novice.values()]
    centroids_e_expert = [v for v in centroids_e_expert.values()]
    centroids_i_expert = [v for v in centroids_i_expert.values()]

    color = []
    for name in names_centroids_expert:
        if 'E_0' in name:
            color.append(0)
        elif 'M' in name:
            color.append(1)
        elif 'E' in name:
            color.append(2)
        elif 'N' in name:
            color.append(3)

    if seniority == 'all':
        # TODO modify colors
        return centroids_e_all, centroids_i_all, names_centroids_all, color

    elif seniority == 'novice':
        # TODO modify colors
        return centroids_e_novice, centroids_i_novice, names_centroids_novice, color

    else:
        return centroids_e_expert, centroids_i_expert, names_centroids_expert, color


def pairwise_centroids_plot_data(errors, ious, names, ids, experts, novices, names_dict):
    centroids_e_all = {id: [] for id in ids}
    centroids_i_all = {id: [] for id in ids}
    centroids_e_expert = {id: [] for id in ids}
    centroids_i_expert = {id: [] for id in ids}
    centroids_e_novice = {id: [] for id in ids}
    centroids_i_novice = {id: [] for id in ids}

    for e, i, n in zip(errors, ious, names):
        i1, i2 = eval(n)
        if i1 in experts or i1 in novices:
            centroids_e_all[i2].append(e)
            centroids_i_all[i2].append(i)
            if i1 in experts:
                centroids_e_expert[i2].append(e)
                centroids_i_expert[i2].append(i)
            if i1 in novices:
                centroids_e_novice[i2].append(e)
                centroids_i_novice[i2].append(i)

        if i2 in experts or i2 in novices:
            centroids_e_all[i1].append(e)
            centroids_i_all[i1].append(i)
            if i2 in experts:
                centroids_e_expert[i1].append(e)
                centroids_i_expert[i1].append(i)
            if i2 in novices:
                centroids_e_novice[i1].append(e)
                centroids_i_novice[i1].append(i)

    for key, value in centroids_e_all.items():
        centroids_e_all[key] = sum(value) / len(value)
    for key, value in centroids_i_all.items():
        centroids_i_all[key] = sum(value) / len(value)
    for key, value in centroids_e_novice.items():
        centroids_e_novice[key] = sum(value) / len(value)
    for key, value in centroids_i_novice.items():
        centroids_i_novice[key] = sum(value) / len(value)
    for key, value in centroids_e_expert.items():
        centroids_e_expert[key] = sum(value) / len(value)
    for key, value in centroids_i_expert.items():
        centroids_i_expert[key] = sum(value) / len(value)

    names_centroids_all = [names_dict[i] for i in ids]
    names_centroids_novice = [names_dict[i] for i in ids]
    names_centroids_expert = [names_dict[i] for i in ids]
    color_centroids_all = [0 for i in ids]
    color_centroids_novice = [1 for i in ids]
    color_centroids_expert = [2 for i in ids]

    centroids_e_all = [v for v in centroids_e_all.values()]
    centroids_i_all = [v for v in centroids_i_all.values()]
    centroids_e_novice = [v for v in centroids_e_novice.values()]
    centroids_i_novice = [v for v in centroids_i_novice.values()]
    centroids_e_expert = [v for v in centroids_e_expert.values()]
    centroids_i_expert = [v for v in centroids_i_expert.values()]

    e_merged = centroids_e_all + centroids_e_novice + centroids_e_expert
    i_merged = centroids_i_all + centroids_i_novice + centroids_i_expert
    n_merged = names_centroids_all + names_centroids_novice + names_centroids_expert
    c_merged = color_centroids_all + color_centroids_novice + color_centroids_expert

    e_merged = centroids_e_all + centroids_e_novice + centroids_e_expert
    i_merged = centroids_i_all + centroids_i_novice + centroids_i_expert
    n_merged = names_centroids_all + names_centroids_novice + names_centroids_expert
    c_merged = color_centroids_all + color_centroids_novice + color_centroids_expert

    return e_merged, i_merged, n_merged, c_merged

def pairwise_signed_pvalues(test_ds_path, model_path):

    with open(model_path, 'r') as f:
        model = json.load(f)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    evaluate_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    experts = [1, 2, 3, 4, 5]
    novices = [6, 7, 8]
    names_dict = {1: "$E_0$", 2: "$E_1$", 3: "$E_2$", 4: "$E_3$", 5: "$E_4$", 6: "$N_1$", 7: "$N_2$", 8: "$N_3$",
                  0: "$M$"}
    data_dict = {0: model}
    names, ious, errors = generate_data(str(test_ds_path), ids, data_dict, evaluate_ids, iou_threshold=0.0,
                                        per_img=True)
    names = [eval(n) for n in names]
    errors = np.array(errors)
    ious = np.array(ious)

    errors_pvals = np.ones((9, 9))
    ious_pvals = np.ones((9, 9))

    for k in range(9):
        for l in range(9):
            if k == l:
                continue
            experts_to_use = [e for e in experts if (e != k and e != l)]

            errors_per_image = np.zeros((100, ))
            ious_per_image = np.zeros((100, ))
            for e in experts_to_use:
                pair1 = (k, e) if k < e else (e, k)
                pair2 = (l, e) if l < e else (e, l)
                pair1_idx = names.index(pair1)
                pair2_idx = names.index(pair2)

                errors_per_image += errors[pair1_idx] - errors[pair2_idx]
                ious_per_image -= ious[pair1_idx] - ious[pair2_idx]
                # errors_per_image[k, l] += errors[pair1_idx] - errors[pair2_idx]
                # ious[k, l] -= ious[pair1_idx] - ious[pair2_idx]

            errors_pvals[k, l] = wilcoxon(errors_per_image).pvalue
            ious_pvals[k, l] = wilcoxon(ious_per_image).pvalue

    names, ious, errors = generate_data(str(test_ds_path), ids, data_dict, evaluate_ids, iou_threshold=0.0,
                                        per_img=False)
    e_merged, i_merged, n_merged, c_merged = pairwise_centroids_plot_data_per_seniority('experts', errors, ious,
                                                                                        names, ids, experts, novices,
                                                                                        names_dict)

    errors_signs = np.ones((9, 9))
    ious_signs = np.ones((9, 9))

    for k in range(9):
        for l in range(9):
            if k == l:
                continue
            if e_merged[k] > e_merged[l]:
                errors_signs[k, l] = -1

            if i_merged[k] < i_merged[l]:
                ious_signs[k, l] = -1
    signed_iou_pvals = np.multiply(1 - ious_pvals, ious_signs)
    signed_error_pvals = np.multiply(1 - errors_pvals, errors_signs)
    return signed_iou_pvals, signed_error_pvals
