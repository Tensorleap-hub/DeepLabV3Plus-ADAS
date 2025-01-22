from code_loader import leap_binder
from code_loader.contract.enums import DatasetMetadataType
from code_loader.contract.datasetclasses import PreprocessResponse
import tensorflow as tf
from PIL import Image
from code_loader.contract.enums import (
    LeapDataType
)

from domain_gap.data.cs_data import Cityscapes, CATEGORIES
from domain_gap.utils.gcs_utils import _download
from domain_gap.tl_helpers.preprocess import subset_images
from domain_gap.tl_helpers.visualizers.visualizers import image_visualizer, loss_visualizer, mask_visualizer, \
    cityscape_segmentation_visualizer
from domain_gap.tl_helpers.utils import get_categorical_mask, get_metadata_json, class_mean_iou, mean_iou
from domain_gap.utils.config import CONFIG
import numpy as np
import os
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess, tensorleap_input_encoder, tensorleap_metadata, )

# ----------------------------------- Input ------------------------------------------

@tensorleap_input_encoder('non_normalized')
def non_normalized_input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx % data["real_size"]]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(CONFIG['IMAGE_SIZE'])) / 255.
    return img.astype(np.float32)

@tensorleap_input_encoder('normalized_image')
def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_input_image(idx % data.data["real_size"], data)
    if data.data['dataset'][idx % data.data["real_size"]] == 'kitti':
        img = (img - CONFIG['KITTI_MEAN']) * CONFIG['CITYSCAPES_STD'] / CONFIG['KITTI_STD'] + CONFIG['CITYSCAPES_MEAN']
    normalized_image = (img - CONFIG['IMAGE_MEAN']) / CONFIG['IMAGE_STD']
    return normalized_image.astype(np.float32)


# ----------------------------------- GT ------------------------------------------

def ground_truth_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    return tf.keras.utils.to_categorical(mask, num_classes=20).astype(float)[...,
           :19]  # Remove background class from cross-entropy


# ----------------------------------- Metadata ------------------------------------------

@tensorleap_metadata("idx")
def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    """ add TL index """
    return idx

@tensorleap_metadata("class_percent")
def metadata_class_percent(idx: int, data: PreprocessResponse) -> dict:
    res = {}
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    for i, c in enumerate(CATEGORIES + ["background"]):
        count_obj = unique_per_obj.get(float(i))
        if count_obj is not None:
            percent_obj = count_obj / mask.size
        else:
            percent_obj = 0.0
        res[f'{c}'] = percent_obj
    return res

def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
    img = non_normalized_input_image(idx % data.data["real_size"], data)
    return np.mean(img)

@tensorleap_metadata("filename")
def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx]

@tensorleap_metadata("city")
def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]

@tensorleap_metadata("dataset")
def metadata_dataset(idx: int, data: PreprocessResponse) -> str:
    return data.data['dataset'][idx % data.data["real_size"]]

@tensorleap_metadata("gps_heading")
def metadata_gps_heading(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsHeading']
    else:
        return CONFIG['DEFAULT_GPS_HEADING']

@tensorleap_metadata("gps_latitude")
def metadata_gps_latitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsLatitude']
    else:
        return CONFIG['DEFAULT_GPS_LATITUDE']

@tensorleap_metadata("gps_longtitude")
def metadata_gps_longtitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsLongitude']
    else:
        return CONFIG['DEFAULT_GPS_LONGTITUDE']

@tensorleap_metadata("outside_temperature")
def metadata_outside_temperature(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['outsideTemperature']
    else:
        return CONFIG['DEFAULT_TEMP']

@tensorleap_metadata("speed")
def metadata_speed(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['speed']
    else:
        return CONFIG['DEFAULT_SPEED']

@tensorleap_metadata("yaw_rate")
def metadata_yaw_rate(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['yawRate']
    else:
        return CONFIG['DEFAULT_YAW_RATE']


def metadata_folder_name(idx: int, data: PreprocessResponse) -> str:
    return os.path.dirname(data.data['paths'][idx])
# ----------------------------------- Binding ------------------------------------------

leap_binder.add_prediction('seg_mask', CATEGORIES)
