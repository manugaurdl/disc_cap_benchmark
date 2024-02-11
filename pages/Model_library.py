import streamlit as st
from io import StringIO
import pandas as pd
import sys
from models.clip import *
from models.blip import *
from PIL import Image
from utils import *



st.set_page_config(page_title="Katha AI Model Library", layout="wide", page_icon="💻", menu_items={
    'About': 'https://github.com/katha-ai',
    'Get Help': 'https://github.com/katha-ai',
    'Report a bug': "mailto:hardik.mittal@research.iiit.ac.in "
})
st.title("KATHA AI's Model Library")
st.subheader("🚧🚧 Page under construction 🚧🚧")
st.subheader("Please use Image Models and Video Models for now")


# uploaded_file = None


# pic_col, vid_col = st.tabs(["Image", "Video"])

# # print(pic_col, vid_col)

# with pic_col:

#     uploaded_file = st.file_uploader("Choose an image", type=[
#         'png', 'jpg'], accept_multiple_files=False,  help='Upload an image')

#     if uploaded_file is not None:
#         left_col, right_col = st.columns(2, gap="large")
#         with left_col:
#             with st.spinner('Loading the image'):
#                 st.image(uploaded_file, caption='Uploaded Image',
#                      use_column_width=True)
#         with right_col:

#             st.subheader("Select the model to run the image on")

#             ALIGN_col, CLIP_col, BLIP_col, ALBEF_col, BLIPv2_col, GLIP_col = st.tabs(
#                 ["ALIGN", "CLIP", "BLIP", "ALBEF", "BLIPv2", "GLIP"])


#             with ALIGN_col:
#                 print("hello align")
#                 # clear cache from ada for all other models by checking if they are in the memory first
#                 # 
#                 st.write('Sending the image to ALIGN model')
#             with CLIP_col:
#                 CLIP_wrapper(uploaded_file)

#             with BLIP_col:
#                 BLIP_wrapper(uploaded_file)

#             with ALBEF_col:
#                 st.write('Sending the image to ALBEF model')
#             with BLIPv2_col:
#                 st.write('Sending the image to BLIPv2 model')
#             with GLIP_col:
#                 st.write('Sending the image to GLIP model')

# with vid_col:
#     uploaded_file = st.file_uploader("Choose a video", type=[
#         'mp4'], accept_multiple_files=False,  help='Upload a video')

#     if uploaded_file is not None:
#         left_col, right_col = st.columns(2, gap="large")
#         with left_col:
#             with st.spinner('Loading the video'):
#                 st.video(uploaded_file)
#         with right_col:
#             st.subheader("Select the model to run the video on")

#             # VideoCLIP, CLIP4Clip, XCLIP, CLIP-ViP, ViFi-CLIP

#             VideoCLIP_col, CLIP4Clip_col, XCLIP_col, CLIP_ViP_col, ViFi_CLIP_col = st.tabs(
#                 ["VideoCLIP", "CLIP4Clip", "XCLIP", "CLIP-ViP", "ViFi-CLIP"])

#             with VideoCLIP_col:
#                 st.write('Sending the video to VideoCLIP model')
#             with CLIP4Clip_col:
#                 st.write('Sending the video to CLIP4Clip model')
#             with XCLIP_col:
#                 st.write('Sending the video to XCLIP model')
#             with CLIP_ViP_col:
#                 st.write('Sending the video to CLIP_ViP model')
#             with ViFi_CLIP_col:
#                 st.write('Sending the video to ViFi_CLIP model')

# # st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
