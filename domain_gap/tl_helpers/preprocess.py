from typing import List, Dict
from code_loader.contract.datasetclasses import PreprocessResponse
import json
from domain_gap.data.cs_data import get_cityscapes_data
from domain_gap.data.kitti_data import get_kitti_data
from domain_gap.utils.config import CONFIG
from os.path import join


def subset_images() -> List[PreprocessResponse]:
    if not CONFIG['USE_LOCAL']:
        subset_sizes = [CONFIG['TRAIN_SIZE'], CONFIG['VAL_SIZE']]
        cs_responses: List[PreprocessResponse] = get_cityscapes_data()
        kitti_data: Dict[str, List[str]] = get_kitti_data()
        sub_names = ["train", "validation"]
        for i, title in enumerate(sub_names):   # add kitti values
            cs_responses[i].data['image_path'] += kitti_data[title]['image_path']
            cs_responses[i].data['gt_path'] += kitti_data[title]['gt_path']
            cs_responses[i].data['file_names'] += kitti_data[title]['image_path']
            cs_responses[i].data['cities'] += ["Karlsruhe"] * len(kitti_data[title]['image_path'])
            cs_responses[i].data['dataset'] += ['kitti'] * len(kitti_data[title]['image_path'])
            cs_responses[i].data['real_size'] += len(kitti_data[title]['image_path'])
            cs_responses[i].data['metadata'] += [""] * len(kitti_data[title]['image_path'])
            cs_responses[i].length += len(kitti_data[title]['image_path'])
            cs_responses[i].length = subset_sizes[i]
    else:
        with open(join(CONFIG['LOCAL_BASE_PATH'], "Cityscapes", "ADAS_preprocess", "train.json"), 'r') as f:
            train_data = json.load(f)
        with open(join(CONFIG['LOCAL_BASE_PATH'], "Cityscapes", "ADAS_preprocess", "val.json"), 'r') as f:
            val_data = json.load(f)
        cs_responses = [PreprocessResponse(data=train_data, length=5), PreprocessResponse(data=val_data, length=5)]
    return cs_responses
