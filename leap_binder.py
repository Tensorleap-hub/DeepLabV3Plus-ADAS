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
from domain_gap.tl_helpers.preprocess import subset_images, unlabeled_images
from domain_gap.tl_helpers.visualizers.visualizers import image_visualizer, loss_visualizer, mask_visualizer, \
    cityscape_segmentation_visualizer, image_visualizer_unnorm
from domain_gap.tl_helpers.utils import get_categorical_mask, get_metadata_json, class_mean_iou, mean_iou, per_class_percentage, row_index
from domain_gap.utils.config import CONFIG
import numpy as np
import os
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess, tensorleap_input_encoder, tensorleap_metadata, tensorleap_gt_encoder,
    tensorleap_custom_loss, )

# ----------------------------------- Input ------------------------------------------

@tensorleap_input_encoder('non_normalized',channel_dim=-1)
def non_normalized_input_image(sample_id: str, data: PreprocessResponse) -> np.ndarray:
    cloud_path = str(data.data['image_path'][row_index(sample_id, data)])
    fpath = cloud_path if os.path.exists(cloud_path) else _download(cloud_path)
    img = np.array(Image.open(fpath).convert('RGB').resize(CONFIG['IMAGE_SIZE'])) / 255.
    return img.astype(np.float32)

@tensorleap_input_encoder('normalized_image',channel_dim=-1)
def input_image(sample_id: str, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_input_image(sample_id, data)
    if data.data['dataset'][row_index(sample_id, data)] == 'kitti':
        img = (img - CONFIG['KITTI_MEAN']) * CONFIG['CITYSCAPES_STD'] / CONFIG['KITTI_STD'] + CONFIG['CITYSCAPES_MEAN']
    normalized_image = (img - CONFIG['IMAGE_MEAN']) / CONFIG['IMAGE_STD']
    return normalized_image.astype(np.float32)


# ----------------------------------- GT ------------------------------------------

@tensorleap_gt_encoder("mask")
def ground_truth_mask(sample_id: str, data: PreprocessResponse) -> np.ndarray:
    if not data.data['gt_path'][row_index(sample_id, data)]:
        return np.array([], dtype=np.float32)
    mask = get_categorical_mask(sample_id, data)
    return tf.keras.utils.to_categorical(mask, num_classes=20).astype(float)[...,:19].astype(np.float32)  # Remove background class from cross-entropy


# ----------------------------------- Metadata ------------------------------------------

@tensorleap_metadata("idx")
def metadata_idx(sample_id: str, data: PreprocessResponse) -> int:
    """ add TL index """
    return row_index(sample_id, data)

@tensorleap_metadata("class_percent")
def metadata_class_percent(sample_id: str, data: PreprocessResponse) -> dict:
    if not data.data['gt_path'][row_index(sample_id, data)]:
        return {f'{c}': None for c in CATEGORIES + ["background"]}
    res = {}
    mask = get_categorical_mask(sample_id, data)
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

def metadata_brightness(sample_id: str, data: PreprocessResponse) -> float:
    img = non_normalized_input_image(sample_id, data)
    return np.mean(img)

@tensorleap_metadata("filename")
def metadata_filename(sample_id: str, data: PreprocessResponse) -> str:
    idx = row_index(sample_id, data)
    return data.data['file_names'][idx]

@tensorleap_metadata("city")
def metadata_city(sample_id: str, data: PreprocessResponse) -> str:
    idx = row_index(sample_id, data)
    return data.data['cities'][idx]

@tensorleap_metadata("dataset")
def metadata_dataset(sample_id: str, data: PreprocessResponse) -> str:
    return data.data['dataset'][row_index(sample_id, data)]

@tensorleap_metadata("gps_heading")
def metadata_gps_heading(sample_id: str, data: PreprocessResponse) -> float:
    idx = row_index(sample_id, data)
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(sample_id, data).get('gpsHeading', CONFIG['DEFAULT_GPS_HEADING'])
    else:
        return CONFIG['DEFAULT_GPS_HEADING']

@tensorleap_metadata("gps_latitude")
def metadata_gps_latitude(sample_id: str, data: PreprocessResponse) -> float:
    idx = row_index(sample_id, data)
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(sample_id, data).get('gpsLatitude', CONFIG['DEFAULT_GPS_LATITUDE'])
    else:
        return CONFIG['DEFAULT_GPS_LATITUDE']

@tensorleap_metadata("gps_longtitude")
def metadata_gps_longtitude(sample_id: str, data: PreprocessResponse) -> float:
    idx = row_index(sample_id, data)
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(sample_id, data).get('gpsLongitude', CONFIG['DEFAULT_GPS_LONGTITUDE'])
    else:
        return CONFIG['DEFAULT_GPS_LONGTITUDE']

@tensorleap_metadata("outside_temperature")
def metadata_outside_temperature(sample_id: str, data: PreprocessResponse) -> float:
    idx = row_index(sample_id, data)
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(sample_id, data).get('outsideTemperature', CONFIG['DEFAULT_TEMP'])
    else:
        return CONFIG['DEFAULT_TEMP']

@tensorleap_metadata("speed")
def metadata_speed(sample_id: str, data: PreprocessResponse) -> float:
    idx = row_index(sample_id, data)
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(sample_id, data).get('speed', CONFIG['DEFAULT_SPEED'])
    else:
        return CONFIG['DEFAULT_SPEED']

@tensorleap_metadata("yaw_rate")
def metadata_yaw_rate(sample_id: str, data: PreprocessResponse) -> float:
    idx = row_index(sample_id, data)
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(sample_id, data).get('yawRate', CONFIG['DEFAULT_YAW_RATE'])
    else:
        return CONFIG['DEFAULT_YAW_RATE']


@tensorleap_custom_loss("categorical_cross_entropy")
def custom_loss(ground_truth: np.array, prediction: np.array) -> np.array:
    ground_truth = ground_truth.astype(np.float32)
    prediction = prediction.astype(np.float32)
    ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    ls_batch = ls(ground_truth, prediction)
    valid = tf.cast(tf.reduce_sum(ground_truth, axis=-1) > 0, ls_batch.dtype)
    ls_image = tf.reduce_sum(ls_batch * valid, axis=[1, 2]) / tf.maximum(tf.reduce_sum(valid, axis=[1, 2]), 1.0)
    return ls_image.numpy()

leap_binder.leap_analysis_configuration.domain_gap_metadata = ['city']

if __name__ == "__main__":
    leap_binder.check()
