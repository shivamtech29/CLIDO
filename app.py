import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image,ImageOps
from keras.preprocessing import image

st.set_page_config(page_title="CLIDO - SA")

#Loading the Model
model = load_model('LumpyDisease.h5', compile=False)

#st.image('logo.png')
st.markdown("<h1 style='text-align:center;color:orange;'>Cam Based Lumpy Infection Detection By Observation : CLIDO</h1>",unsafe_allow_html=True)
st.markdown("<h3 style='text-align:right;'>Created By : SHIVAM AGARWAL</h3>",unsafe_allow_html=True)

file = st.file_uploader("Upload a cow image...", type=['png','jpg','webp','jpeg'])


def predictimg(imagedata,model):
    size = (150,150)
    image = ImageOps.fit(imagedata,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    a = model.predict(img_reshape)
    indices = a.argmax()
    if indices==0:
        ans = 'This is Probably a Healthy cow'
    else:
        ans = 'This is Probably an Infected cow'
    return ans

if file is not None:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    ans = predictimg(image,model)
    st.subheader(ans)
