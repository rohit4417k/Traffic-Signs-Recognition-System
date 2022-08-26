
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import streamlit as st


@st.cache(allow_output_mutation=True)
def get_model():
    model = load_model('final_model.hdf5')
    print('Model Loaded')
    return model


def predict(image):
    loaded_model = get_model()
    image = load_img(image, target_size=(32, 32), color_mode="grayscale")
    image = img_to_array(image)
    image = image / 255.0
    image = np.reshape(image, [1, 32, 32, 1])

    classes = loaded_model.predict(image)

    return classes