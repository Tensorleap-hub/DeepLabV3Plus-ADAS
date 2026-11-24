"""Export models/DeeplabV3.h5 to ONNX (models/DeeplabV3.onnx) for the ONNX Runtime engine.

The Keras model stores its depthwise convolutions as grouped Conv2D layers (groups == channels).
tf2onnx does not export grouped Conv2D, so each one is rebuilt as the equivalent DepthwiseConv2D
first (same weights, kernel axes swapped), then the model is converted with tf2onnx.

Usage (needs tensorflow and tf2onnx; onnxruntime is optional and only used for the parity check):
    pip install tf2onnx==1.16.1 onnxruntime
    python models/export_onnx.py [--h5 models/DeeplabV3.h5] [--out models/DeeplabV3.onnx]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import tensorflow as tf

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def _to_depthwise(layer):
    cfg = layer.get_config()
    is_grouped = (isinstance(layer, tf.keras.layers.Conv2D)
                  and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)
                  and cfg.get("groups", 1) > 1)
    if not is_grouped:
        return layer.__class__.from_config(cfg)
    channels = layer.input_shape[-1]
    if not cfg["groups"] == channels == cfg["filters"]:
        raise ValueError(f"{layer.name}: grouped conv is not depthwise "
                         f"(groups={cfg['groups']}, in={channels}, out={cfg['filters']})")
    return tf.keras.layers.DepthwiseConv2D(
        kernel_size=cfg["kernel_size"], strides=cfg["strides"], padding=cfg["padding"],
        dilation_rate=cfg["dilation_rate"], use_bias=cfg["use_bias"],
        activation=cfg["activation"], name=cfg["name"])


def grouped_to_depthwise(model):
    new = tf.keras.models.clone_model(model, clone_function=_to_depthwise)
    swapped = 0
    for old_layer, new_layer in zip(model.layers, new.layers):
        weights = old_layer.get_weights()
        if (isinstance(new_layer, tf.keras.layers.DepthwiseConv2D)
                and not isinstance(old_layer, tf.keras.layers.DepthwiseConv2D)):
            weights[0] = np.transpose(weights[0], (0, 1, 3, 2))  # (kh, kw, 1, C) -> (kh, kw, C, 1)
            swapped += 1
        new_layer.set_weights(weights)
    return new, swapped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", default=os.path.join(MODELS_DIR, "DeeplabV3.h5"))
    parser.add_argument("--out", default=os.path.join(MODELS_DIR, "DeeplabV3.onnx"))
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.h5, compile=False)
    fixed, swapped = grouped_to_depthwise(model)
    x = np.random.RandomState(0).rand(1, *model.input_shape[1:]).astype(np.float32)
    keras_out = model(x).numpy()
    diff = np.abs(keras_out - fixed(x).numpy()).max()
    print(f"grouped Conv2D -> DepthwiseConv2D: {swapped} layers, max abs diff {diff:.2e}")

    with tempfile.TemporaryDirectory() as saved_model_dir:
        fixed.save(saved_model_dir, save_format="tf")
        subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", saved_model_dir,
                        "--opset", str(args.opset), "--output", args.out], check=True)
    print(f"wrote {args.out}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping the parity check")
        return
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: x})[0]
    print(f"ONNX Runtime vs Keras: max abs diff {np.abs(onnx_out - keras_out).max():.2e}")


if __name__ == "__main__":
    main()
