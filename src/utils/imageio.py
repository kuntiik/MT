__all__ = [
    "ImgSize",
    "open_img",
    "get_img_size",
]

from PIL import ExifTags
from collections import namedtuple
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List, Any, Union

ImgSize = namedtuple("ImgSize", "width,height")
#
# # get exif tag
for _EXIF_ORIENTATION_TAG in ExifTags.TAGS.keys():
    if PIL.ExifTags.TAGS[_EXIF_ORIENTATION_TAG] == "Orientation":
        break


def open_img(fn, gray=False, ignore_exif: bool = False) -> PIL.Image.Image:
    "Open an image from disk `fn` as a PIL Image"
    color = "L" if gray else "RGB"

    image = PIL.Image.open(str(fn))

    if not ignore_exif:
        image = PIL.ImageOps.exif_transpose(image)
    image = image.convert(color)
    return image


def get_img_size(filepath: Union[str, Path]) -> ImgSize:
    """
    Returns image (width, height)
    """
    image = PIL.Image.open(str(filepath))
    image_size = image.size

    try:
        exif = image._getexif()
        if exif is not None and exif[_EXIF_ORIENTATION_TAG] in [6, 8]:
            image_size = image_size[::-1]
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return ImgSize(*image_size)
