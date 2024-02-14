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
class Benchmark:
    def __init__(self):
        self.ann = None
        self.caption_type = None 
        self.argsorted_dist_matrix_ids = None
        self.cocoid2cap = None
        self.idx2cocoid = None 
        self.bag_size = None
        self.bag_idx = None

    def init_data(self):

        idx2cocoid = open_json(os.path.join(data_dir, "coco_test_val_idx2cocoid.json")) # this comes from test_val_cocoids. Invariant to caption type
        
        if self.caption_type == "coco":
            cocoid2cap = open_pickle("/home2/manugaur/synthetic_data/data/cocoid2caption.pkl")
            
        if multimodal:
            filename = f"clip_mm_{self.caption_type}"
            folder = "mm_feats"  # for argsorted_dist_matrix_ids
        ann_filepath = os.path.join(data_dir, "ann", filename + ".npy")
        ann = np.load(ann_filepath)
        argsorted_dist_matrix_ids = np.load(os.path.join(data_dir, folder, "argsorted_dist_matrix_ids", filename + f"_bsz_{self.bag_size}.npy"))

        st.write("data loaded")
        return ann, argsorted_dist_matrix_ids, idx2cocoid, cocoid2cap

    def on_callback(self):
        st.write("in callback")
        get_bags(self.ann, self.argsorted_dist_matrix_ids, self.idx2cocoid, bag_idx = self.bag_idx, bag_size= self.bag_size, cocoid2cap = self.cocoid2cap, align = ALIGN)

    def main(self):
        # if 'bag_idx' not in st.session_state:
        #     st.session_state.bag_idx = 0
        # if 'caption_type' not in st.session_state:
        #     st.session_state.caption_type = None

        
        caption_type = st.selectbox("Select caption type", ["coco", "mistral"])
        
        if self.caption_type != caption_type:
            self.caption_type = caption_type
            # SELECT BAG_SIZE   
            # if 'bag_size' not in st.session_state:
            #     st.session_state.bag_size = 3
            bag_size = st.number_input("Select bag size : ", 2, 10, 3, key="new_bag_size")

            if self.bag_size != bag_size:
                self.bag_size = bag_size
    
            self.ann, self.argsorted_dist_matrix_ids, self.idx2cocoid, self.cocoid2cap = self.init_data()
            st.write(self.ann.shape)


        # SELECT BAG_IDX 
            
        bag_idx = st.number_input("Select bag idx to retrieve : ", 0, 9999, 0, key="new_bag_idx")
            
        if self.bag_idx != bag_idx:
            self.bag_idx = bag_idx
            # self.on_callback(self.ann, self.argsorted_dist_matrix_ids, self.idx2cocoid, self.cocoid2cap)
            self.on_callback()     
        
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
    obj = Benchmark()
    obj.main()