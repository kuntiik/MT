__all__ = [
    "Adapter",
    "AlbumentationsAdapterComponent",
    "AlbumentationsImgComponent",
    "AlbumentationsSizeComponent",
    "AlbumentationsInstancesLabelsComponent",
    "AlbumentationsBBoxesComponent",
]

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Callable, List

import albumentations as A
from itertools import chain

import numpy as np

from mt.core import BBox, BaseRecord
from mt.core.composite import Component, Composite
from mt.transforms.albumentations_utils import get_size_without_padding
from mt.utils import ImgSize


@dataclass
class CollectOp:
    fn: Callable
    order: float = 0.5


class Transform(ABC):
    def __call__(self, record: BaseRecord):
        # TODO: this assumes record is already loaded and copied
        # which is generally true
        return self.apply(record)

    @abstractmethod
    def apply(self, record: BaseRecord) -> BaseRecord:
        """Apply the transform
        Returns:
              dict: Modified values, the keys of the dictionary should have the same
              names as the keys received by this function
        """


class AlbumentationsAdapterComponent(Component):
    @property
    def adapter(self):
        return self.composite

    def setup(self):
        return

    def prepare(self, record):
        pass

    def collect(self, record):
        pass


class AlbumentationsImgComponent(AlbumentationsAdapterComponent):
    def setup_img(self, record):
        # NOTE - assumed that `record.img` is a PIL.Image
        self.adapter._albu_in["image"] = np.array(record.img)

        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        record.set_img(self.adapter._albu_out["image"])


class AlbumentationsSizeComponent(AlbumentationsAdapterComponent):
    order = 0.2

    def setup_size(self, record):
        self.adapter._collect_ops.append(CollectOp(self.collect, order=0.2))

    def collect(self, record) -> ImgSize:
        # return self._size_no_padding
        width, height = self.adapter._size_no_padding
        record.set_image_size(width=width, height=height)


class AlbumentationsInstancesLabelsComponent(AlbumentationsAdapterComponent):
    order = 0.1

    def set_labels(self, record, labels):
        # TODO HACK: Will not work for multitask, will fail silently
        record.detection.set_labels_by_id(labels)

    def setup_instances_labels(self, record_component):
        # TODO HACK: Will not work for multitask, will fail silently
        self._original_labels = record_component.label_ids
        # Substitue labels with list of idxs, so we can also filter out iscrowds in case any bboxes are removed
        self.adapter._albu_in["labels"] = list(range(len(self._original_labels)))

        self.adapter._collect_ops.append(CollectOp(self.collect_labels, order=0.1))

    def collect_labels(self, record):
        self.adapter._keep_mask = np.zeros(len(self._original_labels), dtype=bool)
        self.adapter._keep_mask[self.adapter._albu_out["labels"]] = True

        labels = self.adapter._filter_attribute(self._original_labels)
        self.set_labels(record, labels)


class AlbumentationsBBoxesComponent(AlbumentationsAdapterComponent):
    def setup_bboxes(self, record_component):
        self.adapter._compose_kwargs["bbox_params"] = A.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        )
        # TODO: albumentations has a way of sending information that can be used for tasks

        # TODO HACK: Will not work for multitask, will fail silently
        self.adapter._albu_in["bboxes"] = [o.xyxy for o in record_component.bboxes]

        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record) -> List[BBox]:
        # TODO: quickfix from 576
        # bboxes_xyxy = [_clip_bboxes(xyxy, img_h, img_w) for xyxy in d["bboxes"]]
        bboxes_xyxy = [xyxy for xyxy in self.adapter._albu_out["bboxes"]]
        bboxes = [BBox.from_xyxy(*xyxy) for xyxy in bboxes_xyxy]
        # TODO HACK: Will not work for multitask, will fail silently
        record.detection.set_bboxes(bboxes)

    @staticmethod
    def _clip_bboxes(xyxy, h, w):
        """Clip bboxes coordinates that are outside image dimensions."""
        x1, y1, x2, y2 = xyxy
        if w >= h:
            pad = (w - h) // 2
            h1 = pad
            h2 = w - pad
            return (x1, max(y1, h1), x2, min(y2, h2))
        else:
            pad = (h - w) // 2
            w1 = pad
            w2 = h - pad
            return (max(x1, w1), y1, min(x2, w2), y2)


class AlbumentationsAreasComponent(AlbumentationsAdapterComponent):
    def setup_areas(self, record_component):
        self._areas = record_component.areas
        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        areas = self.adapter._filter_attribute(self._areas)
        record.detection.set_areas(areas)


class Adapter(Transform, Composite):
    base_components = {
        AlbumentationsImgComponent,
        AlbumentationsSizeComponent,
        AlbumentationsInstancesLabelsComponent,
        AlbumentationsBBoxesComponent,
        AlbumentationsAreasComponent,
    }

    def __init__(self, tfms):
        super().__init__()
        self.tfms_list = tfms

    def create_tfms(self):
        return A.Compose(self.tfms_list, **self._compose_kwargs)

    def apply(self, record):
        # setup
        self._compose_kwargs = {}
        self._keep_mask = None
        self._albu_in = {}
        self._collect_ops = []
        record.setup_transform(tfm=self)

        # TODO: composing every time
        tfms = self.create_tfms()
        # apply transform
        self._albu_out = tfms(**self._albu_in)

        # store additional info (might be used by components on `collect`)
        height, width, _ = self._albu_out["image"].shape
        height, width = get_size_without_padding(
            self.tfms_list, record.img, height, width
        )
        self._size_no_padding = ImgSize(width=width, height=height)

        # collect results
        for collect_op in sorted(self._collect_ops, key=lambda x: x.order):
            collect_op.fn(record)

        return record

    def _filter_attribute(self, v: list):
        if self._keep_mask is None or len(self._keep_mask) == 0:
            return v
        assert len(v) == len(self._keep_mask)
        return [o for o, keep in zip(v, self._keep_mask) if keep]


def _flatten_tfms(t):
    flat = []
    for o in t:
        if _is_iter(o):
            flat += [i for i in o]
        else:
            flat.append(o)
    return flat


def _is_iter(o):
    try:
        i = iter(o)
        return True
    except:
        return False
