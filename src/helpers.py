import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from src.constants import IM_HEIGHT, IM_WIDTH

def load_img(path):
    """
    Reading an image with tf including resizing and padding
    """
    img = tf.io.read_file(path)

    # grayscale
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = IM_HEIGHT / tf.cast(h, tf.float32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img = tf.image.resize(img, (IM_HEIGHT, new_w))

    # pad or squeeze
    curr_w = tf.shape(img)[1]
    if curr_w > IM_WIDTH:
        img = tf.image.resize(img, (IM_HEIGHT, IM_WIDTH))
    else:
        pad_w = IM_WIDTH - curr_w
        img = tf.pad(img, [[0, 0], [0, pad_w], [0, 0]], constant_values=0.)
    return img


def decode_true_labels(ds, decode_fn):
    truths = []
    for _, labels in ds:
        for seq in labels.numpy():
            seq = seq[seq != 0]
            truths.append(b"".join(decode_fn(seq).numpy()).decode("utf-8"))
    return truths


def decode_model_output(y_pred, decode_fn):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_len, greedy=True)
    res = []
    for seq in decoded[0].numpy():
        seq = seq[seq != -1]
        text = b"".join(decode_fn(seq).numpy()).decode("utf-8")
        res.append(text)
    return res


def build_dataset(paths, labels, batch_size, encode_fn):
    """
    For testing only
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_and_process(p, y):
        img = load_img(p)
        y = encode_fn(tf.strings.unicode_split(y, input_encoding="UTF-8"))
        y = tf.cast(y, tf.int32) + 1
        return {"img_input": img, "label_input": y}, y

    ds = ds.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(batch_size, padded_shapes=({"img_input": [IM_HEIGHT, IM_WIDTH, 1], "label_input": [None]}, [None]),
        padding_values=({"img_input": 0.0, "label_input": 0}, 0)
    )
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
    