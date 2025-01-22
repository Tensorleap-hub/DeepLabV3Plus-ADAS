from typing import List, Dict, Union
from code_loader.contract.datasetclasses import PreprocessResponse
import json
from domain_gap.data.cs_data import get_cityscapes_data
from domain_gap.data.kitti_data import get_kitti_data
from domain_gap.utils.config import CONFIG
from os.path import join
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess

@tensorleap_preprocess()
def subset_images() -> List[PreprocessResponse]:
    if not CONFIG['USE_LOCAL']:
        cs_dicts: List[Dict[str, Union[str, float, int]]] = get_cityscapes_data()
        kitti_data: Dict[str, List[str]] = get_kitti_data()
        sub_names = ["train", "validation"]
        length_addition = [0, 0]
        for i, title in enumerate(sub_names):   # add kitti values
            cs_dicts[i]['image_path'] += kitti_data[title]['image_path']
            cs_dicts[i]['gt_path'] += kitti_data[title]['gt_path']
            cs_dicts[i]['file_names'] += kitti_data[title]['image_path']
            cs_dicts[i]['cities'] += ["Karlsruhe"] * len(kitti_data[title]['image_path'])
            cs_dicts[i]['dataset'] += ['kitti'] * len(kitti_data[title]['image_path'])
            cs_dicts[i]['real_size'] += len(kitti_data[title]['image_path'])
            cs_dicts[i]['metadata'] += [""] * len(kitti_data[title]['image_path'])
            length_addition[i] += len(kitti_data[title]['image_path'])
        if CONFIG['OVERRIDE_SIZE']:
            sizes = [CONFIG['TRAIN_SIZE'], CONFIG['VAL_SIZE']]
        else:
            sizes = [len(cs_dicts[i]['image_path']) + length_addition[i] for i in range(2)]
        cs_responses = [PreprocessResponse(length=sizes[i], data=cs_dicts[i]) for i in range(len(cs_dicts))]
    else:
        with open(join(CONFIG['LOCAL_BASE_PATH'], "Cityscapes", "ADAS_preprocess", "train.json"), 'r') as f:
            train_data = json.load(f)
        with open(join(CONFIG['LOCAL_BASE_PATH'], "Cityscapes", "ADAS_preprocess", "val.json"), 'r') as f:
            val_data = json.load(f)
        cs_responses = [PreprocessResponse(data=train_data, length=5), PreprocessResponse(data=val_data, length=5)]
    return cs_responses
