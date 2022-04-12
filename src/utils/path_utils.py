__all__ = ["get_data_dir", "get_root_dir"]

from pathlib import Path

root_dir = Path.home() / ".mt"
root_dir.mkdir(exist_ok=True)

data_dir = root_dir / "data"
data_dir.mkdir(exist_ok=True)


def get_data_dir():
    return data_dir


def get_root_dir():
    return root_dir
