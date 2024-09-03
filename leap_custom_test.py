import urllib
from os.path import exists
from leap_binder import *

def check_custom_integration():
    print("statedtesting")
    # loading model
    model_path = 'models/DeeplabV3.h5'
    if not exists(model_path):
        print("Downloading DeeplabV3.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/domain_gap/DeeplabV3.h5",
            model_path)
    model = tf.keras.models.load_model(model_path)
    idx = 0
    # gathering data
    responses = subset_images()  # get dataset splits
    train_res = responses[0]  # [training, validation, test]
    # get plot images
    image = input_image(idx, train_res)  # get specific image
    image_visualizer(image).plot_visualizer()
    # get plot gt
    mask_gt = ground_truth_mask(idx, train_res)  # get image gt
    cityscape_segmentation_visualizer(mask_gt).plot_visualizer()
    input_img = np.expand_dims(image, axis=0)
    # predict
    y_pred = model([input_img])[0] # infer and get model prediction
    # compute loss visalize
    loss_visualizer_img = loss_visualizer(image, y_pred, mask_gt)
    loss_visualizer_img.plot_visualizer()
    # custom metrics
    class_iou_res = class_mean_iou(mask_gt, y_pred)
    print(f"Metics: class_mean_iou - {class_iou_res}")
    metric_result = mean_iou(y_pred, mask_gt)
    print(f"Metics: mean_iou - {metric_result}")
    # print metadata
    for metadata_handler in leap_binder.setup_container.metadata:
        curr_metadata = metadata_handler.function(idx, train_res)
        print(f"Metadata {metadata_handler.name}: {curr_metadata}")


if __name__ == "__main__":
    check_custom_integration()