import numpy as np
import numpy.typing as npt
import tensorflow as tf
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapImage, LeapImageMask
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer

from domain_gap.data.cs_data import Cityscapes
from domain_gap.tl_helpers.visualizers.visualizers_utils import unnormalize_image, scalarMap
from domain_gap.data.cs_data import CATEGORIES


@tensorleap_custom_visualizer("image_visualizer", LeapDataType.Image)
def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    image = np.squeeze(image)
    return LeapImage((unnormalize_image(image) * 255).astype(np.uint8))

@tensorleap_custom_visualizer("image_visualizer_unnorm", LeapDataType.Image)
def image_visualizer_unnorm(image: npt.NDArray[np.float32]) -> LeapImage:
    image = np.squeeze(image)
    return LeapImage((image * 255).astype(np.uint8))

@tensorleap_custom_visualizer("mask_visualizer", LeapDataType.ImageMask)
def mask_visualizer(image: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]) -> LeapImageMask:
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    excluded_mask = mask.sum(-1) == 0
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)
    mask[excluded_mask] = 19
    return LeapImageMask(mask.astype(np.uint8), unnormalize_image(image).astype(np.float32), CATEGORIES + ["excluded"])

@tensorleap_custom_visualizer("cityscapes_visualizer", LeapDataType.Image)
def cityscape_segmentation_visualizer(mask: npt.NDArray[np.uint8]) -> LeapImage:
    mask = np.squeeze(mask)
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            cat_mask = np.squeeze(mask, axis=-1)
        else:
            cat_mask = np.argmax(mask, axis=-1)  # this introduce 0 at places where no GT is present (zero all channels)
    else:
        cat_mask = mask
    cat_mask[mask.sum(-1) == 0] = 19  # this marks the place with all zeros using idx 19
    mask_image = Cityscapes.decode_target(cat_mask)
    return LeapImage(mask_image.astype(np.uint8))

@tensorleap_custom_visualizer("loss_visualizer", LeapDataType.Image)
def loss_visualizer(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32], gt: npt.NDArray[np.float32]) -> LeapImage:
    image = np.squeeze(image)
    prediction = np.squeeze(prediction)
    gt = np.squeeze(gt)
    image = unnormalize_image(image)
    ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    ls_image = ls(gt, prediction).numpy()
    ls_image = ls_image.clip(0, np.percentile(ls_image, 95))
    ls_image /= ls_image.max()
    heatmap = scalarMap.to_rgba(ls_image)[..., :-1]
    overlayed_image = ((heatmap * 0.4 + image * 0.6).clip(0,1)*255).astype(np.uint8)
    # overlayed_image = ((heatmap).clip(0, 1) * 255).astype(np.uint8)
    return LeapImage(overlayed_image)


