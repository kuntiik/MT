from pathlib import Path
import PIL
from typing import Union


def end2end_detect(
        img : Union[PIL.Image.Image, Path, str]
):

    if isinstance(img, (str, Path)):
        img = PIL.Image.open(Path(img))
    return None
