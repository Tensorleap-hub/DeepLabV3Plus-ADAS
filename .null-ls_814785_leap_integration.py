import urllib
from os.path import exists

import numpy as np

from leap_binder import *
from pathlib import Path

from code_loader import LeapLoader
from code_loader.plot_functions.visualize import visualize
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test


prediction_type1 = PredictionTypeHandler('seg_mask', CATEGORIES)

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'models/DeeplabV3.h5'
    if not exists(model_path):
        print("Downloading DeeplabV3.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/domain_gap/DeeplabV3.h5",
            model_path)
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return cnn

@tensorleap_integration_test()
def check_custom_integration(idx, subset):
    print("stated testing")
    plot = True
    model = load_model()
    # get plot images
    image = input_image(idx, subset)  # get specific image
    # image = non_normalized_input_image(idx, subset)  # get specific image
    # get gt
    mask_gt = ground_truth_mask(idx, subset)  # get image gt
    # predict
    y_pred = model([image])  # infer and get model prediction

    # vis
    image_visualizer_ = image_visualizer(image)
    un_norm_image_vis = image_visualizer_unnorm(image)

    cityscape_segmentation_visualizer_ = cityscape_segmentation_visualizer(mask_gt)
    mask_gt_vis = mask_visualizer(image, mask_gt)
    mask_pred_vis = mask_visualizer(image, y_pred)
    loss_visualizer_img = loss_visualizer(image, y_pred, mask_gt)
    ls = custom_loss(mask_gt, y_pred)
    if plot:
        visualize(image_visualizer_, "Input Image")
        visualize(un_norm_image_vis, "Input Image Unnorm")
        visualize(cityscape_segmentation_visualizer_)
        visualize(mask_gt_vis, "GT Mask")
        visualize(mask_pred_vis, "Predicted Mask")
        visualize(loss_visualizer_img, "Loss Vis")

    # custom metrics
    metric_result = mean_iou(mask_gt, y_pred)
    print(f"Metics: mean_iou - {metric_result}")
    class_iou_res = class_mean_iou(mask_gt, y_pred)
    print(f"Metics: class_mean_iou - {class_iou_res}")
    class_percent_res = per_class_percentage(mask_gt, y_pred)
    print(f"Metics: per_class_percentage - {class_percent_res}")
    # print metadata
    for metadata_func in (metadata_idx, metadata_class_percent, metadata_gps_heading, metadata_gps_latitude,
                          metadata_gps_longtitude, metadata_outside_temperature, metadata_speed, metadata_yaw_rate):
        print(metadata_func(idx, subset))


if __name__ == "__main__":
    responses = subset_images()  # get dataset splits
    for subset in responses:
        for i in range(5):
            check_custom_integration(i, subset)
