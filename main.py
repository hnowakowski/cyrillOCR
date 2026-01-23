import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="keras.src.export.tf2onnx_lib"
)
import streamlit as st
from src.helpers import load_img_bytes, decode_model_output, load_MY_model, get_encode_funs, build_dataset
import tensorflow as tf

@st.cache_resource
def setup():
    label_to_int, label_to_str = get_encode_funs()
    model = load_MY_model()
    print("Model loaded")
    return label_to_int, label_to_str, model

if __name__ == "__main__":
    label_to_int, label_to_str, model = setup()

    img_raw = st.file_uploader("lol", accept_multiple_files=False, type="png")
    if img_raw is not None:
        img = load_img_bytes(img_raw.read())
        img = tf.expand_dims(img, axis=0)
        preds = model.predict(img)
        decoded = decode_model_output(preds, label_to_str)
        st.write(decoded[0])
        st.image(img_raw)
    