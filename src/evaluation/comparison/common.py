from typing import List
import  numpy as np

class Centroid:
    def __init__(self, box, coco_format=False):
        self.box = box
        self.coco_format = coco_format
        if coco_format:
            x1, y1, w, h = box
            self.x = (x1 + w/2)
            self.y = (y1 + h/2)
        else:
            x1, y1, x2, y2 = box
            self.x = (x1 + x2) / 2
            self.y = (y1 + y2) / 2

    def inside(self, box) -> bool:
        if self.coco_format:
            x1, y1, w, h = box
            if x1 <= self.x <= x1 + w and y1 <= self.y <= y1 + h:
                return True
            return False
        else:
            x1, y1, x2, y2 = box
            if x1 <= self.x <= x2 and y1 <= self.y <= y2:
                return True
            return False

    def inside_batch(self, array) -> List[bool]:
        return  [self.inside(box) for box in array]

    def other_centroids(self, array):
        return [Centroid(b, self.coco_format).inside(self.box) for b in array]

    def match_criterion(self, array):
        a = self.inside_batch(array)
        b = self.other_centroids(array)
        return np.logical_or(a, b)