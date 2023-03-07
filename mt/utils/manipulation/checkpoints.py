import yaml
from pathlib import Path
import shutil


def get_backbone(b: str):
    b = b if type(b) == str else str(b)
    if 'swin_t' in b:
        return 'swin_t'
    if 'resnet50' in b:
        return 'resnet50'
    if 'resnet101' in b:
        return 'resnet101'
    if 'small' in b:
        return 'small'
    if 'medium' in b:
        return 'medium'
    if 'extra_large' in b:
        return 'extra_large'
    if 'large' in b:
        return 'large'
    if 'tf_d0' in b:
        return 'tf_d0'
    if 'tf_d1' in b:
        return 'tf_d1'
    if 'tf_d2' in b:
        return 'tf_d2'
    if 'tf_d3' in b:
        return 'tf_d3'
    if 'tf_d4' in b:
        return 'tf_d4'
    if 'tf_d5' in b:
        return 'tf_d5'
    if 'd0' in b:
        return 'd0'
    if 'd1' in b:
        return 'd1'
    if 'd2' in b:
        return 'd2'
    if 'd3' in b:
        return 'd3'
    if 'd4' in b:
        return 'd4'
    if 'd5' in b:
        return 'd5'


def get_architecture(a: str):
    a = a if type(a) == str else str(a)
    if 'yolov5' in a:
        return 'yolov5'
    if 'faster_rcnn' in a:
        return 'faster_rcnn'
    if 'retinanet' in a:
        return 'retinanet'
    if 'efficientdet' in a:
        return 'efficientdet'


def arch_bb(cfg):
    architecture_key = 'model/model/model/_target_'
    backbone_key = 'model/model/model/backbone/_target_'
    gn_key = 'model/group_norm'
    if not architecture_key in cfg.keys() or not backbone_key in cfg.keys():
        return None
    else:
        architecture = get_architecture(cfg[architecture_key]['value'])
        backbone = get_backbone(cfg[backbone_key]['value'])
        #check if group norm was used
        if gn_key in cfg.keys() and cfg[gn_key]['value']:
            backbone = 'gn_' + backbone
        return architecture, backbone


def load_cfg(path):
    cfg_path = path / 'wandb/latest-run/files/config.yaml'
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return None


def load_ckpt(path):
    ckpt_path = path / 'checkpoints'
    if not ckpt_path.exists():
        return None
    else:
        ckpts = list(ckpt_path.glob('epoch*.ckpt'))
        if len(ckpts) == 1:
            return ckpts[0]
        else:
            return None


def set_name(ckpt_path):
    name = ckpt_path.stem
    name = name.split('_')[-1] + ckpt_path.suffix
    while ckpt_path.with_name(name).exists():
        name = name.replace(ckpt_path.suffix, '') + '0' + ckpt_path.suffix
    return ckpt_path.with_name(name)


def get_cfgs_ckpt(ckpt_folder):
    ckpt_folder = Path(ckpt_folder) if type(ckpt_folder) is str else ckpt_folder
    configs = []
    checkpoints = []
    for path in ckpt_folder.iterdir():
        if (path / 'multirun.yaml').exists():
            for p in path.iterdir():
                if not p.is_dir():
                    continue
                configs.append(load_cfg(p))
                checkpoints.append(load_ckpt(p))
        else:
            configs.append(load_cfg(path))
            checkpoints.append(load_ckpt(path))
    return configs, checkpoints


def organize_ckpts(ckpt_folder: Path, target_folder):
    configs, checkpoints = get_cfgs_ckpt(ckpt_folder)

    for cfg, ckpt in zip(configs, checkpoints):
        if cfg is None or ckpt is None:
            continue
        else:
            ret_val = arch_bb(cfg)
            if ret_val is None:
                return
            arch, bb = ret_val
            target = target_folder / arch / bb
            target.mkdir(parents=True, exist_ok=True)
            name = set_name(target / ckpt.name)

            shutil.copyfile(str(ckpt), name)

def move_ckpts(ckpt_folder: Path, target_folder):
    for ckpt in ckpt_folder.rglob("*.ckpt"):
        arch = get_architecture(ckpt)
        bb = get_backbone(ckpt)
        (target_folder / arch / bb).mkdir(parents=True, exist_ok=True)
        ckpt_name = target_folder / arch / bb / ckpt.name
        name = ckpt.name
        while ckpt_name.with_name(name).exists():
            name = name.replace(ckpt_name.suffix, '') + '0' + ckpt_name.suffix
        ckpt_name = ckpt_name.with_name(name)
        shutil.copyfile(str(ckpt), ckpt_name)


