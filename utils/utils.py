import streamlit as st
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import gc
import os
import clip
import torch
import random
from tqdm import tqdm
import sys
import torch.nn.functional as F
import faiss
import sys
import math

def get_cocoid_list(split : str):
    split_cocoids_path = f"/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/data/parse_coco_req/{split}_cocoids.json"
    with open(split_cocoids_path, "rb") as f: 
        return json.load(f)

def open_json(path : str):
    with open(path, "r") as f:
        return json.load(f)

def open_pickle(path : str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_data(multimodal):
    st.write("loading data")
    # test + val cocoids
    test_val_cocoids = open_json("/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/data/parse_coco_req/test_val_cocoids.json")
    # Load CLIP feats
    clip_feat_path = "/ssd_scratch/cvit/manu/img_cap_self_retrieval_clip/data/clip_feats"

    cocoid2idx = open_json(os.path.join(clip_feat_path, "coco_test_val_cocoid2idx.json"))

    clip_text_feats = F.normalize(torch.load(os.path.join(clip_feat_path,"coco_test_val_text_avg.pt")), p=2, dim =-1)
    clip_img_feats = F.normalize(torch.load(os.path.join(clip_feat_path,"coco_test_val_img_avg.pt")), p=2, dim =-1)

    # Get distance matrix 
    # Define feat 
    mm_clip_feats = clip_img_feats + clip_text_feats
    if multimodal:
        feats = mm_clip_feats.clone()    
    return cocoid2idx, feats

def build_dist_matrix(cosine_sim, feats):
    st.write("calculating dist matrix")
    feats = feats.cpu().numpy().astype(np.float32)
    # st.write(f"clip feats shape : {feats.shape}")

    clip_dim = feats.shape[1]
    if cosine_sim: 
        index = faiss.IndexFlatIP(clip_dim)
    else:
        index = faiss.IndexFlatL2(clip_dim)
    faiss.normalize_L2(feats)
    index.add(feats)

    search_vector = feats
    # st.write(f"search vector shape : {search_vector.shape}")

    faiss.normalize_L2(search_vector)
    k = index.ntotal
    st.write(f"Creating bags across {k} cocoids")
    distances, ann = index.search(search_vector, k=k) # ann --> indices corresponding to distances.

    # First retrieval is cocoid itself
    return ann[:, 1:], distances[:, 1:]

def cocoid2img(cocoid : int, only_path : bool = False):
    
    # print(f"cocoid : {cocoid}")
    
    img_path = f"/ssd_scratch/cvit/manu/coco/train2014/COCO_train2014_{int(cocoid):012d}.jpg"
    if not os.path.isfile(img_path):
        img_path = f"/ssd_scratch/cvit/manu/coco/val2014/COCO_val2014_{int(cocoid):012d}.jpg"
    if only_path:
        return img_path
    image = Image.open(img_path)
    image = image.resize((300, 300))
    st.image(image, caption = "image")
    # image = io.imread(img_path)
    # image = Image.fromarray(image)
    # st.write('\n')
    
    # fig = plt.figure(figsize=(3,3))
    # plt.imshow(image)
    # plt.axis('off')
    # # plt.show()
    # st.pyplot()


def quali_analysis(cocoid : int, plot : bool, preds : bool, gt : bool):
    if plot:
        cocoid2img(cocoid)
    st.write(f"cocoid : {cocoid}")
    if preds :
        st.write(f"\ncvpr23 : {cvpr23[cocoid]}")
        st.write(f"cider : {cvpr_cider[cocoid]}\n ")

        st.write(f"cider  : {cider_optim[cocoid]}")
        st.write(f"cider : {cider_optim_cider[cocoid]}\n ")

        st.write(f"mle    : {clipcap_mle[cocoid]}")
        st.write(f"cider : {mle_cider[cocoid]} ")

    if gt :     
        st.write(f"BLIP:\n{blip[cocoid]}")

        st.write(f"\nllama GT : {cocoid2llama_gt[cocoid]} \n")

        st.write("narrations:")
        for narration in narrations[cocoid]:
            st.write(narration)
        st.write("\nCOCO GT:")
        for cap in cocoid2cap[cocoid]:
            st.write(cap) 

def get_bags(ann, argsorted_dist_matrix_ids, idx2cocoid,  bag_idx : int, bag_size : int, cocoid2cap, align : bool):
    """
    get N bags with highest cumsum (until k retrievals) cosine sim 
    """
    st.write({f"bag_idx = {bag_idx}"})
    bag = []
    seen_imgs = set()
    clip_idx = argsorted_dist_matrix_ids[bag_idx]
    cocoid_1 = idx2cocoid[clip_idx]
    bag.append(cocoid_1)
    # quali_analysis(cocoid_1, plot = True, preds = False, gt = False)

    for idx, sim_img in enumerate(ann[clip_idx][:bag_size - 1]):
        cocoid_2 = idx2cocoid[sim_img]
        bag.append(cocoid_2)
        # quali_analysis(cocoid_2, plot = True, preds = False, gt = False)
            
        # st.write("*" * 300)
        # st.write("\n")
    if not align :
        max_cols = 3
        num_cols = min(bag_size, max_cols)
        cols = st.columns(num_cols)
        
        for idx, cocoid in enumerate(bag):

            col_idx = idx  % num_cols
            image_file = cocoid2img(cocoid, only_path = True)
            
            with Image.open(image_file) as image:
                cols[col_idx].image(image, caption = f"{idx}",use_column_width=True)
                cols[col_idx].write(f"This is a cap")
    else:
        max_cols = 3
        num_cols = min(bag_size, max_cols)
        cols = st.columns(num_cols)
        
        max_height = 0
        image_heights = []
        for idx, cocoid in enumerate(bag):

            col_idx = idx  % num_cols
            image_file = cocoid2img(cocoid, only_path = True)
            
            with Image.open(image_file) as image:
                cols[col_idx].image(image, caption = f"{cocoid}",use_column_width=True)
                image_height = image.height
                image_heights.append(image_height)
                max_height = max(max_height, image_height)
            
            if col_idx == max_cols - 1 or idx == bag_size -1 :

                for idx, h in enumerate(image_heights):
                    diff = math.ceil((max_height -h)//80)
                    for M in range(diff):
                        cols[idx].write("\n")
                    cols[idx].write(cocoid2cap[int(cocoid)])
                    # for cap in cocoid2cap[int(cocoid)]:
                    #     cols[idx].write(f"<span style='font-size: 12px; color : green;line-height: 0;'>{cap}</span>", unsafe_allow_html=True)
                image_heights = []
                max_height = 0             