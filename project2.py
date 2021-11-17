import streamlit as st 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import streamlit as st
import requests
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img
# from tensorflow.keras.preprocessing import image
from keras import backend as K
from tensorflow import keras
from keras.models import load_model

img_height, img_width  = 180, 180


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
def load_image(image_file):

	img = Image.open(image_file)
	return img 


# def predict_image(filename):
#     img = load_img(filename, target_size=(img_height, img_width))
#     image = keras.preprocessing.image.img_to_array(img)
#     image = image / 255.0
#     image = image.reshape(1,180,180,3)
#     model = load_model('chestxray_cnn_model_3.h5')
#     prediction = model.predict(image)
#     plt.imshow(img)
#     if(prediction[0] > 0.5):
#         stat = prediction[0] * 100 
#         st.write("This image is %.2f percent %s"% (stat, "PNEUMONIA"))
#     else:
#         stat = (1.0 - prediction[0]) * 100
#         st.write("This image is %.2f percent %s" % (stat, "NORMAL"))

st.title("Convoloution NN using Keras to predict pneumonia")
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.sidebar.subheader("Visualization setup")
uploaded_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
try:
    with open(uploaded_file.name,"wb") as f:
        f.write(uploaded_file.getbuffer())
except AttributeError:
    pass
# st.success["File Saved"]

# file_details = {"Filename":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
# st.write(file_details)

if not uploaded_file:
    st.warning("Please upload an image before proceeding!")
    st.stop()
else:
    img = load_img(uploaded_file.name, target_size=(img_height, img_width))
    image = keras.preprocessing.image.img_to_array(img)
    image = image / 255.0
    image = image.reshape(1,180,180,3)
    model = load_model('chestxray_cnn_model_3.h5')
    prediction = model.predict(image)
    plt.imshow(img)
    if(prediction[0] > 0.5):
        stat = prediction[0] * 100 
        st.title("This image is %.2f percent %s"% (stat, "PNEUMONIA"))
    else:
        stat = (1.0 - prediction[0]) * 100
        st.title("This image is %.2f percent %s" % (stat, "NORMAL"))
    image_as_bytes = uploaded_file.read()
    st.image(image_as_bytes, use_column_width=True)
    # a = "'"+image_as_bytes+"'"
    # predict_image(a)
    

