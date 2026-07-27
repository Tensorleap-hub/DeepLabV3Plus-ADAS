from typing import List, Dict, Union
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DataStateType
import json
from domain_gap.data.cs_data import get_cityscapes_data
from domain_gap.data.kitti_data import get_kitti_data
from domain_gap.utils.config import CONFIG
from os.path import join
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess, tensorleap_unlabeled_preprocess


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
        with open(join(CONFIG['LOCAL_BASE_PATH'], "original_csv_subset", "train.json"), 'r') as f:
            train_data = json.load(f)
        with open(join(CONFIG['LOCAL_BASE_PATH'], "original_csv_subset", "val.json"), 'r') as f:
            val_data = json.load(f)
        cs_responses = [PreprocessResponse(data=train_data, length=train_data["real_size"]),
                         PreprocessResponse(data=val_data, length=val_data["real_size"])]
    return cs_responses


@tensorleap_unlabeled_preprocess()
def unlabeled_images() -> PreprocessResponse:
    """Mock unlabeled set: cityscapes samples keep real GT (loaded from the full local
    Cityscapes mirror), raw KITTI drive frames stay genuinely unlabeled (no GT)."""
    with open(join(CONFIG['LOCAL_BASE_PATH'], "unlabeled_subset", "manifest.json"), 'r') as f:
        manifest = json.load(f)

    image_path, gt_path, gt_image_path = [], [], []
    file_names, cities, dataset, metadata = [], [], [], []

    for sample in manifest:
        image_path.append(sample["path"])
        file_names.append(sample["filename"])
        metadata.append("")

        if sample["dataset"] == "cityscapes":
            city, split = sample["city"], sample["split"]
            stem = sample["filename"].replace("_leftImg8bit.png", "")
            gt_path.append(join(CONFIG['UNLABELED_CITYSCAPES_GT_ROOT'], "gtFine_trainvaltest", "gtFine", split, city,
                                 f"{stem}_gtFine_labelIds.png"))
            gt_image_path.append(join(CONFIG['UNLABELED_CITYSCAPES_GT_ROOT'], "gtFine_trainvaltest", "gtFine", split, city,
                                       f"{stem}_gtFine_color.png"))
            cities.append(city)
            dataset.append("cityscapes")
        else:
            gt_path.append("")
            gt_image_path.append("")
            cities.append(sample.get("city", "Karlsruhe"))
            dataset.append("kitti")

    data = {
        "subset_name": "unlabeled",
        "image_path": image_path,
        "gt_path": gt_path,
        "gt_image_path": gt_image_path,
        "file_names": file_names,
        "cities": cities,
        "dataset": dataset,
        "metadata": metadata,
        "real_size": len(image_path),
    }
    return PreprocessResponse(length=len(image_path), data=data, state=DataStateType.unlabeled)
