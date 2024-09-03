import os
from typing import Dict, Any
import yaml
import numpy as np

def load_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'project_config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    turn_to_numpy = ['IMAGE_MEAN', 'IMAGE_STD', 'KITTI_MEAN', 'KITTI_STD', 'CITYSCAPES_MEAN', 'CITYSCAPES_STD']
    for key in turn_to_numpy:
        config[key] = np.array(config[key])
    return config


CONFIG = load_config()
