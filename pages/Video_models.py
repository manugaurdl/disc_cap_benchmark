import streamlit as st
from utils.utils import *
from PIL import Image
import gc

uploaded_file = st.file_uploader("Choose a video", type=[
    'mp4'], accept_multiple_files=False,  help='Upload a video')

if uploaded_file is not None:
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        with st.spinner('Loading the video'):
            st.video(uploaded_file)
    with right_col:
        st.subheader("Select the model to run the video on")

        # VideoCLIP, CLIP4Clip, XCLIP, CLIP-ViP, ViFi-CLIP

        VideoCLIP_col, CLIP4Clip_col, XCLIP_col, CLIP_ViP_col, ViFi_CLIP_col = st.tabs(
            ["VideoCLIP", "CLIP4Clip", "XCLIP", "CLIP-ViP", "VIFICLIP"])

        with VideoCLIP_col:
            # st.write('Sending the video to VideoCLIP model')
            choose_vid_task(uploaded_file, "VideoCLIP")            
        with CLIP4Clip_col:
            choose_vid_task(uploaded_file, "CLIP4CLIP")            
        with XCLIP_col:
            choose_vid_task(uploaded_file, "XCLIP")            
        with CLIP_ViP_col:
            st.write('Sending the video to CLIP_ViP model')
        with ViFi_CLIP_col:
            choose_vid_task(uploaded_file, "VIFICLIP")            

# st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])

