# export PATH="$HOME/.local/bin:$PATH"
import streamlit as st
import gc
import clip
import os
import numpy as np
import torch
import random
from tqdm import tqdm
import sys
import faiss
from evaluate_nlg import *
from utils.utils import *

gc.enable()
device = "cuda" if torch.cuda.is_available() else "cpu"
multimodal = True
cosine_sim = True
np.set_printoptions(precision = 2, suppress=True)
ALIGN = True
# BAG_IDX = 0 
st.set_page_config(
    page_title="Discriminant Captioning Benchmark",
    page_icon="",
    layout = "wide"
)

st.write("# Discriminant Captioning Benchmark")


def init_data(caption_type):
    if caption_type =="coco":
        cocoid2idx, feats = load_data(multimodal)
        idx2cocoid = {v: k for k,v in cocoid2idx.items()}
        ann, distances = build_dist_matrix(cosine_sim, feats)
        cocoid2cap = open_pickle("/home2/manugaur/synthetic_data/data/cocoid2caption.pkl")
    return ann, distances, idx2cocoid, cocoid2cap

def on_callback(ann, idx2cocoid, cocoid2cap,  align):
    get_bags(ann, st.session_state.argsorted_dist_matrix_ids, idx2cocoid, bag_idx = st.session_state.bag_idx, bag_size= st.session_state.bag_size, cocoid2cap = cocoid2cap, align = ALIGN)

# def get_next_bag(ann, distances, idx2cocoid, align):
#     get_bags(ann, distances, idx2cocoid, bag_idx = st.session_state.bag_idx, bag_size= st.session_state.bag_size, align = ALIGN)

def main():
    caption_type = st.selectbox("Select caption type", ["coco", "mistral"])

    if 'caption_type' not in st.session_state:
        st.session_state.prev_caption_type = None
    
    if st.session_state.prev_caption_type != caption_type:
        ann, distances, idx2cocoid, cocoid2cap = init_data(caption_type)
        st.session_state.prev_caption_type = caption_type
        

    # SELECT BAG_SIZE
    if 'bag_size' not in st.session_state:
        st.session_state.bag_size = 3

    bag_size = st.number_input("Select bag size : ", 2, 10, 3, key="new_bag_size")

    if st.session_state.bag_size != bag_size:
        st.session_state.bag_size = bag_size
        cum_sum = np.sum(distances[:, :bag_size -1 ], axis = 1) # shape = num_rows
        argsorted_dist_matrix_ids = np.argsort(cum_sum)[::-1]  #sort clip idx in descending order of cumsum of k most sim ids
        st.session_state.argsorted_dist_matrix_ids = argsorted_dist_matrix_ids
        on_callback(ann, idx2cocoid, cocoid2cap, ALIGN)

    # Select BAG_IDX
    if 'bag_idx' not in st.session_state:
        st.session_state.bag_idx = 0
    
    bag_idx = st.number_input("Select bag idx to retrieve : ", 0, 9999, 0, key="new_bag_idx")
    
    if st.session_state.bag_idx != bag_idx:
        st.session_state.bag_idx = bag_idx
        on_callback(ann, idx2cocoid, cocoid2cap, ALIGN)
    # st.markdown(
    #     """
    #     <style>
    #     .stButton>button {
    #         float: right;
    #         position: fixed;
    #         bottom: 20px;
    #         right: 20px;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    #st.button('Next', on_click = get_next_bag, args = (ann, distances, idx2cocoid, ALIGN, ))

if __name__ == "__main__":
    main()