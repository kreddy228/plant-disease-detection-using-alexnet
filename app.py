import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow import keras
from time import time

st.set_option('deprecation.showfileUploaderEncoding', False)

model = tf.keras.models.load_model('net_model.hdf5')

st.write("""
         # ***Plant Disease Classifier***
         """
         )

st.write("Plant Disease Classifier web-app")

file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])

li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

if file is None:
    st.text("You haven't uploaded a jpg image file")
else:
    classifier = keras.models.load_model('net_model.hdf5')
    imageI = Image.open(file)
    image_path = "history/" + f"{int(time())}.JPG" # Image.open(file)
    with open(image_path, "wb") as f:
        f.write(file.getbuffer())
    new_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    print("Following is our prediction:")
    prediction = classifier.predict(img)
    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index]
    
    st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write(f"""
                 ## **Prediction :** {class_name}!!
                 """
                 )