# export PATH="$HOME/.local/bin:$PATH"
import streamlit as st
# import gc
# import clip
import os
import numpy as np
# import torch
# import random
# from tqdm import tqdm
# import sys
# import faiss
# from evaluate_nlg import *
from utils.utils import open_json, open_pickle, get_bags

# gc.enable()
# device = "cuda" if torch.cuda.is_available() else "cpu"
multimodal = True
np.set_printoptions(precision = 2, suppress=True)
data_dir = "/ssd_scratch/cvit/manu/eccv_benchmark"
ALIGN = False

# BAG_IDX = 0 
st.set_page_config(
    page_title="Discriminant Captioning Benchmark",
    page_icon="",
    layout = "wide"
)

st.write("# Discriminant Captioning Benchmark")

def init_data(caption_type, multimodal, bag_size = 3):

    idx2cocoid = open_json(os.path.join(data_dir, "coco_test_val_idx2cocoid.json")) # this comes from test_val_cocoids. Invariant to caption type
    
    if caption_type == "coco":
        cocoid2cap = open_pickle("/home2/manugaur/synthetic_data/data/cocoid2caption.pkl")
        
    if multimodal:
        filename = f"clip_mm_{caption_type}"
        folder = "mm_feats"  # for argsorted_dist_matrix_ids
    ann_filepath = os.path.join(data_dir, "ann", filename + ".npy")
    ann = np.load(ann_filepath)
    argsorted_dist_matrix_ids = np.load(os.path.join(data_dir, folder, "argsorted_dist_matrix_ids", filename + f"_bsz_{bag_size}.npy"))

    st.write("data loaded")
    return ann, argsorted_dist_matrix_ids, idx2cocoid, cocoid2cap

def on_callback(ann, argsorted_dist_matrix_ids, idx2cocoid, cocoid2cap):
    get_bags(ann, argsorted_dist_matrix_ids, idx2cocoid, bag_idx = st.session_state.bag_idx, bag_size= 3, cocoid2cap = cocoid2cap, align = ALIGN)


# def get_next_bag(ann, distances, idx2cocoid, align):
#     get_bags(ann, distances, idx2cocoid, bag_idx = st.session_state.bag_idx, bag_size= st.session_state.bag_size, align = ALIGN)

def main():
    caption_type = st.selectbox("Select caption type", ["coco", "mistral"])

    if 'caption_type' not in st.session_state:
        st.session_state.prev_caption_type = None
    
    if st.session_state.prev_caption_type != caption_type:
        ann, argsorted_dist_matrix_ids, idx2cocoid, cocoid2cap = init_data(caption_type, multimodal)
        st.session_state.prev_caption_type = caption_type
        

    # # SELECT BAG_SIZE
    # if 'bag_size' not in st.session_state:
    #     st.session_state.bag_size = 3

    # bag_size = st.number_input("Select bag size : ", 2, 10, 3, key="new_bag_size")

    # if st.session_state.bag_size != bag_size:
    #     st.session_state.bag_size = bag_size
    #     cum_sum = np.sum(distances[:, :bag_size -1 ], axis = 1) # shape = num_rows
    #     argsorted_dist_matrix_ids = np.argsort(cum_sum)[::-1]  #sort clip idx in descending order of cumsum of k most sim ids
    #     st.session_state.argsorted_dist_matrix_ids = argsorted_dist_matrix_ids
    #     on_callback(ann, idx2cocoid, cocoid2cap, ALIGN)

    # Select BAG_IDX

    if 'bag_idx' not in st.session_state:
        st.session_state.bag_idx = 0
    
    bag_idx = st.number_input("Select bag idx to retrieve : ", 0, 9999, 0, key="new_bag_idx")
    
    if st.session_state.bag_idx != bag_idx:
        st.session_state.bag_idx = bag_idx
        on_callback(ann, argsorted_dist_matrix_ids, idx2cocoid, cocoid2cap)
    # # st.markdown(
    # #     """
    # #     <style>
    # #     .stButton>button {
    # #         float: right;
    # #         position: fixed;
    # #         bottom: 20px;
    # #         right: 20px;
    # #     }
    # #     </style>
    # #     """,
    # #     unsafe_allow_html=True
    # # )
    # #st.button('Next', on_click = get_next_bag, args = (ann, distances, idx2cocoid, ALIGN, ))

if __name__ == "__main__":
    main()