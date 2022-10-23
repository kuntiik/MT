from pathlib import Path
from typing import Union
import json
import numpy as np


def load_dict(path: Union[Path, str, dict]):
    """If the path argument is dict does nothing, else loads the dict from the json specified by path."""
    if type(path) == str or isinstance(path, Path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return path

def np_matrix_to_latex_table(table, decimals=2):
    str_table = []
    table = np.round(table, decimals=decimals)
    for row in table:
        str_row = []
        for c in row:
            if c % 1 == 0:
                str_row.append(str(int(c)))
            else:
                str_row.append(str(c))
        str_table.append(' & '.join(str_row))
    return str_table