import urllib
from os.path import exists
from leap_binder import *

def check_custom_integration():
    print("statedtesting")
    model_path = 'models/DeeplabV3.h5'
    if not exists(model_path):
        print("Downloading DeeplabV3.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/domain_gap/DeeplabV3.h5",
            model_path)
    model = tf.keras.models.load_model(model_path)
    idx = 0
    responses = subset_images()  # get dataset splits
    train_res = responses[0]  # [training, validation, test]
    image = input_image(idx, train_res)  # get specific image
    mask_gt = ground_truth_mask(idx, train_res)  # get image gt
    y_true_masks = tf.expand_dims(tf.convert_to_tensor(mask_gt), 0)  # convert ground truth bbs to tensor
    input_img = np.expand_dims(image, axis=0)
    input_img_tf = tf.convert_to_tensor(input_img)
    y_pred = model([input_img])[0] # infer and get model prediction
    # visualizers
    loss_visualizer_img = loss_visualizer(image, y_pred, mask_gt)
    class_iou_res = class_mean_iou(mask_gt, y_pred)
    # custom metrics
    metric_result = mean_iou(y_pred, mask_gt)

    class_percent = metadata_class_percent(idx, train_res)
    filename = metadata_filename(idx, train_res)
    city = metadata_city(idx, train_res)
    dataset = metadata_dataset(idx, train_res)
    idx = metadata_idx(idx, train_res)
    gps_heading = metadata_gps_heading(idx, train_res)
    gps_latitude = metadata_gps_latitude(idx, train_res)
    gps_longtitude = metadata_gps_longtitude(idx, train_res)
    outside_temperature = metadata_outside_temperature(idx, train_res)
    speed = metadata_speed(idx, train_res)
    yaw_rate = metadata_yaw_rate(idx, train_res)


if __name__ == "__main__":
    check_custom_integration()