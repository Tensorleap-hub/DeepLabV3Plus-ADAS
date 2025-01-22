import urllib
from os.path import exists
from leap_binder import *
from pathlib import Path

from code_loader import LeapLoader
from code_loader.helpers import visualize


def check_custom_integration():
    print("stated testing")
    plot = True
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
            if plot:
                image_visualizer_ = image_visualizer(image)
                visualize(image_visualizer_)
            # get plot gt
            mask_gt = ground_truth_mask(idx, subset)  # get image gt
            if plot:
                cityscape_segmentation_visualizer_ = cityscape_segmentation_visualizer(mask_gt)
                visualize(cityscape_segmentation_visualizer_)

            input_img = np.expand_dims(image, axis=0)
            # predict
            y_pred = model([input_img])[0] # infer and get model prediction
            # compute loss visalize
            loss_visualizer_img = loss_visualizer(image, y_pred.numpy(), mask_gt)
            if plot:
                visualize(loss_visualizer_img)
            # custom metrics
            class_iou_res = class_mean_iou(mask_gt, y_pred.numpy())
            print(f"Metics: class_mean_iou - {class_iou_res}")
            metric_result = mean_iou(y_pred.numpy(), mask_gt)
            print(f"Metics: mean_iou - {metric_result}")
            # print metadata
            for metadata_handler in leap_binder.setup_container.metadata:
                curr_metadata = metadata_handler.function(idx, subset)
                print(f"Metadata {metadata_handler.name}: {curr_metadata}")




if __name__ == "__main__":
    check_custom_integration()