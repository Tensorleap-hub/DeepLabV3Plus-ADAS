import urllib
from os.path import exists

import numpy as np

from leap_binder import *
from pathlib import Path

from code_loader import LeapLoader
from code_loader.helpers import visualize


def check_custom_integration():
    print("stated testing")
    plot = False
    check_generic = True
    if check_generic:
        leap_binder.check()
    # loading model
    model_path = 'models/DeeplabV3.h5'
    if not exists(model_path):
        print("Downloading DeeplabV3.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/domain_gap/DeeplabV3.h5",
            model_path)
    model = tf.keras.models.load_model(model_path)
    responses = subset_images()  # get dataset splits
    # gathering data
    for i in range(2):
        subset = responses[i]  # [training, validation, test]
        for j in range(5):
            idx = j
            # get plot images
            image = input_image(idx, subset)  # get specific image

            # get gt
            mask_gt = ground_truth_mask(idx, subset)  # get image gt

            # predict
            input_img = np.expand_dims(image, axis=0)
            y_pred = model([input_img])[0]  # infer and get model prediction

            # vis
            mask_gt_batch = np.expand_dims(mask_gt, 0)
            pred_batch = np.expand_dims(y_pred.numpy(), 0)
            image_visualizer_ = image_visualizer(input_img)
            cityscape_segmentation_visualizer_ = cityscape_segmentation_visualizer(mask_gt_batch)
            mask_gt_vis = mask_visualizer(input_img, mask_gt_batch)
            mask_pred_vis = mask_visualizer(input_img, pred_batch)
            loss_visualizer_img = loss_visualizer(input_img, pred_batch, mask_gt_batch)

            if plot:
                visualize(image_visualizer_, "Input Image")
                visualize(cityscape_segmentation_visualizer_)
                visualize(mask_gt_vis, "GT Mask")
                visualize(mask_pred_vis, "Predicted Mask")
                visualize(loss_visualizer_img, "Loss Vis")

            # custom metrics
            metric_result = mean_iou(mask_gt_batch, pred_batch)
            print(f"Metics: mean_iou - {metric_result}")
            class_iou_res = class_mean_iou(mask_gt_batch, pred_batch)
            print(f"Metics: class_mean_iou - {class_iou_res}")
            # print metadata
            for metadata_handler in leap_binder.setup_container.metadata:
                curr_metadata = metadata_handler.function(idx, subset)
                print(f"Metadata {metadata_handler.name}: {curr_metadata}")


if __name__ == "__main__":
    check_custom_integration()
