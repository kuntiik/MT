__all__ = ["download_mmdet_configs", "mmdet_configs_path"]

import shutil
from pathlib import Path

from src.utils import logger
from src.utils.download_utils import download_and_extract
from src.utils.path_utils import get_root_dir

VERSION = "v2.20.1"
BASE_URL = "https://github.com/airctic/mmdetection_configs/archive/refs/tags"


def download_mmdet_configs() -> Path:
    save_dir = get_root_dir() / f"mmdetection_configs"

    mmdet_config_path = save_dir / f"mmdetection_configs-{VERSION[1:]}/configs"
    download_path = save_dir / f"{VERSION}.zip"

    if mmdet_config_path.exists():
        logger.info(
            f"The mmdet config folder already exists. No need to downloaded it. Path : {mmdet_config_path}"
        )
    elif download_path.exists():
        # The zip file was downloaded by not extracted yet
        # Extract zip file
        logger.info(f"Extracting the {VERSION}.zip file.")
        save_dir = Path(download_path).parent
        shutil.unpack_archive(filename=str(download_path), extract_dir=str(save_dir))
    else:
        save_dir.mkdir(parents=True, exist_ok=True)

        download_path = save_dir / f"{VERSION}.zip"
        if not download_path.exists():
            logger.info("Downloading mmdet configs")
            download_and_extract(f"{BASE_URL}/{VERSION}.zip", download_path)

    return mmdet_config_path


mmdet_configs_path = download_mmdet_configs()
