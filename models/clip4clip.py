from PIL import Image
import streamlit as st
from utils.utils import *
import os
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import cv2
import numpy as np
import tempfile

def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),            
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

def video2image(video_path, frame_rate=1.0, size=224):
 
    
    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32) 
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec*fps, end_sec*fps, interval))
        ret = True     
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)
            
        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES , idx)
            ret, frame = cap.read()    
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             
            last_frame = i
            images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))
            
        images = images[:last_frame+1]
    cap.release()
    video_frames = torch.tensor(images)
    return video_frames

def encode_video(video_input):
    video_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    vide_model = video_model.eval()
    video = video2image(video_input)
    visual_output = video_model(video)

    # Normalizing the embeddings and calculating mean between all embeddings. 
    visual_output = visual_output["image_embeds"]
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    visual_output = torch.mean(visual_output, dim=0)
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    return visual_output

def encode_text(search_sentence):
        tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")

        inputs = tokenizer(text=search_sentence , return_tensors="pt")   #(tokenized sentence, attention mask). Mask is torch.ones, there is no padding.
        outputs = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]) #tuple of 2 embeddings : (512 dim, (k tokens,512) dim)
        # embedding of EOS token at last layer : caption representation
        # Normalize embeddings for retrieval:
        final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
        final_output = final_output.cpu().detach().numpy()
        return final_output


def CLIP4CLIP_classification_model(video, prompt):
    """
    args : video, list of prompts
    returns : prob for each prompt
    """

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())
    #encoding text and vid
    text_emb_list = [encode_text(i) for i in prompt]
    vid_emb = encode_video(tfile.name)

    text_emb = torch.tensor(text_emb_list).squeeze(1)
    sim_vector = text_emb@vid_emb
    # assigning prob to each prompt
    probs = sim_vector.softmax(dim = 0)
    
    return probs.unsqueeze(0)



if __name__ == "__main__":
    vid_input = input("Enter the path to the video: ")

    prompt1 = input("Enter prompt 1: ")
    prompt2 = input("Enter prompt 2: ")
    prompt = [prompt1, prompt2]
    print(CLIP4CLIP_classification_model(vid_input, prompt))
