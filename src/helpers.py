import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from constants import IM_HEIGHT, IM_WIDTH

def load_img(path: str) :
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