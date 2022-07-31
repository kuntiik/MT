from pathlib import Path
from typing import Union
import json


def load_dict(path: Union[Path, str, dict]):
    """If the path argument is dict does nothing, else loads the dict from the json specified by path."""
    if type(path) == str or isinstance(path, Path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return path
