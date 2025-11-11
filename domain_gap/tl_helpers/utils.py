import json
from typing import Dict
import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse
from PIL import Image
import tensorflow as tf
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import confusion_matrix
# from tensorflow.python.ops import math_ops

from code_loader.contract.enums import MetricDirection
from domain_gap.utils.gcs_utils import _download
from domain_gap.data.cs_data import Cityscapes, CATEGORIES
from domain_gap.utils.config import CONFIG
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_metric )

@tensorleap_custom_metric("iou_class", {f'{c}': MetricDirection.Upward for c in CATEGORIES})
def class_mean_iou(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

    Args:
        y_true (np.ndarray):Ground truth segmentation mask array of shape (batch_size, height, width, num_classes).
        y_pred (np.ndarray): Predicted segmentation mask array of shape (batch_size, height, width, num_classes).

    Returns:
        res: Dictionary with the mean IOU for each class, calculated per batch.
    """
    res = {}
    for i, c in enumerate(CATEGORIES):
        y_true_i, y_pred_i = y_true[..., i], y_pred[..., i]
        res[f'{c}'] = mean_iou(y_true_i, y_pred_i)
    return res

# Add percentage of each class in the prediction mask
@tensorleap_custom_metric("per_class_prediction_percentage", compute_insights={f'{c}': False for c in CATEGORIES})
def per_class_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    #calculate percentage while keeping batch dim

    res = {}
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1, y_pred.shape[-1])
    total_pixels = y_pred_flat.shape[1]
    for i, c in enumerate(CATEGORIES):
        y_pred_i = y_pred_flat[..., i]
        sigmoid_pred_i = 1 / (1 + np.exp(-y_pred_i))
        sigmoid_pred_i = (sigmoid_pred_i > 0.5).astype(np.float32)
        class_pixels = np.sum(sigmoid_pred_i, axis=-1)
        percentage = class_pixels / total_pixels * 100.0
        res[f'{c}'] = percentage.astype(np.float32)
    return res


def get_class_mean_iou(class_i: int = None):

    def class_mean_iou(y_true, y_pred):
        """
        Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask tensor.
            y_pred (tf.Tensor): Predicted segmentation mask tensor.

        Returns:
            tf.Tensor: Mean Intersection over Union (mIOU) value.
        """
        y_true, y_pred = y_true[..., class_i], y_pred[..., class_i]
        iou = mean_iou(y_true, y_pred)

        return iou

    return class_mean_iou

@tensorleap_custom_metric("iou", MetricDirection.Upward)
def mean_iou(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

    Args:
        y_true (np.ndarray): Ground truth segmentation mask tensor.
        y_pred (np.ndarray): Predicted segmentation mask tensor.

    Returns:
        np.ndarray: Mean Intersection over Union (mIOU) value.
    """
    # Flatten the tensors
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1).astype(y_true.dtype)
    y_pred_flat = 1 / (1 + np.exp(-y_pred_flat))
    y_pred_bin = (y_pred_flat > 0.5).astype(np.float32)

    # Calculate the intersection and union
    intersection = np.sum(y_true_flat * y_pred_bin, axis=-1)
    union = np.sum(np.maximum(y_true_flat, y_pred_bin), axis=-1)

    # Compute IoU, avoid division by zero
    iou = intersection / union if union > 0 else np.nan
    iou = iou.astype(np.float32)
    return iou

def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_path'][idx % data["real_size"]]
    fpath = _download(cloud_path)
    mask = np.array(Image.open(fpath).resize(CONFIG['IMAGE_SIZE'], Image.Resampling.NEAREST))
    if data['dataset'][idx % data["real_size"]] == 'cityscapes':
        encoded_mask = Cityscapes.encode_target_cityscapes(mask)
    else:
        encoded_mask = Cityscapes.encode_target(mask)
    return encoded_mask


def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str, str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict


def aug_factor_or_zero(idx: int, data: PreprocessResponse, value: float) -> float:
    if data.data["subset_name"] == "train" and CONFIG['AUGMENT'] and idx > CONFIG['TRAIN_SIZE'] - 1:
        return value.numpy()
    else:
        return 0.