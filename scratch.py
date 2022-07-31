from torchvision.ops.boxes import box_iou
import torch

boxes_list = [[
    [0.0, 0.0, 0.5, 0.3],  # cluster 1
    [0.0, 0.0, 0.4, 0.3],  # cluster 1
    [0.5, 0.0, 1, 0.3],  # cluster 2
    [0.5, 0.0, 0.9, 0.3],  # cluster 2
    [0.3, 0.2, 0.6, 0.5],  # cluster 3
    [0.2, 0.2, 0.6, 0.5],  # cluster 3
], [
    [0.05, 0.0, 0.5, 0.3],  # cluster 1
    [0.6, 0.0, 1, 0.3],  # cluster 2
    [0.2, 0.2, 0.7, 0.5],  # cluster 3
]]
print(box_iou(torch.tensor(boxes_list[0]), torch.tensor(boxes_list[1])))

