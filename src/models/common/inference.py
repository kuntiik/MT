from pathlib import Path
from typing import List
from typing import Union

import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core import Prediction
from src.utils.ice_utils import pbar


def end2end_detect(
        img: Union[PIL.Image.Image, Path, str]
):
    if isinstance(img, (str, Path)):
        img = PIL.Image.open(Path(img))
    return None


@torch.no_grad()
def _predict_from_dl(
        predict_fn,
        model: nn.Module,
        infer_dl: DataLoader,
        keep_images: bool = False,
        show_pbar: bool = True,
        **predict_kwargs,
) -> List[Prediction]:
    all_preds = []
    for batch, records in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(
            model=model,
            batch=batch,
            records=records,
            keep_images=keep_images,
            **predict_kwargs,
        )
        all_preds.extend(preds)

    return all_preds
