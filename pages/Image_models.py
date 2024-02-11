import streamlit as st
from utils.utils import *
from PIL import Image
import gc

model=None
processor = None
gc.enable()

st.set_page_config(page_title="Image Model Library", layout="wide", page_icon="💻", menu_items={
    'About': 'https://github.com/katha-ai',
    'Get Help': 'https://github.com/katha-ai',
    'Report a bug': "mailto:hardik.mittal@research.iiit.ac.in "
})

uploaded_file = st.file_uploader("Choose an image", type=[
    'png', 'jpg', 'jpeg'], accept_multiple_files=False,  help='Upload an image')

readme_content='''


## What is this page about?
This is the page for Image Models.

## What are the models available?
The following models are available for the following tasks:
* Image Classification
    - ALIGN
    - CLIP
    - BLIP
    - BLIPv2
    - GLIP

* Image Captioning
    

* Question Answering
    - BLIPv2


# How to use this page?
* Upload an image
* Select the task you want to perform on the image
* Select the model you want to run the image on
* Enter the prompts if required
* Click on the run button
* Wait for the results to load




'''
if uploaded_file is None:
    
    st.code(readme_content, language='markdown')

if uploaded_file is not None:
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        with st.spinner('Loading the image'):
            st.image(uploaded_file, caption='Uploaded Image',
                    use_column_width=True)
    with right_col:

        st.subheader("Select the model to run the image on")

        ALIGN_col, CLIP_col, BLIP_col, ALBEF_col, BLIPv2_col, GLIP_col = st.tabs(
            ["ALIGN", "CLIP", "BLIP", "ALBEF", "BLIPv2", "GLIP"])


        with ALIGN_col:
            choose_task(uploaded_file, "ALIGN")
        with CLIP_col:
            choose_task(uploaded_file, "CLIP")
        with BLIP_col:
            choose_task(uploaded_file,"BLIP")

        with ALBEF_col:
            st.write('Sending the image to ALBEF model')
        with BLIPv2_col:
            choose_task(uploaded_file,"BLIPv2")
        with GLIP_col:
            st.write('Sending the image to GLIP model')
