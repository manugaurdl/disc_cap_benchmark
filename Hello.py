# export PATH="$HOME/.local/bin:$PATH"
import streamlit as st
import gc

gc.enable()

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Katha AI! 👋")



st.markdown(
    """
    Katha AI is a research group led by Makarand Tapaswi at IIIT Hyderabad. 
    We are interested in building machines that can understand the world around them, especially by learning from movies and videos.

    **👈 Select Model Library from the sidebar** to play around with a range of image and video based models.

    ### Want to learn more?
    - Check out [our github](https://github.com/katha-ai)

    ### See Papers from Katha AI
    - How you feelin' ? [Learning Emotions and Mental States in Movie Scenes](https://katha-ai.github.io/projects/emotx/)
    
"""
)