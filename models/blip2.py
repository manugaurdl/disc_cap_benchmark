import requests
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2Model, AutoTokenizer
from utils.utils import *
import streamlit as st

def BLIP2_Question_Answering_Model(image, prompt):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7B")
    
    model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

    model.to(device)
    prompt = "Question: " + prompt + " Answer: " 
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)

    
    outputs = model(**inputs)

    # this is the image-text similarity score
    
    # logits_per_image = outputs.logits_per_image

    # we can take the softmax to get the label probabilities
    # probs = logits_per_image.softmax(dim=1)
    return outputs


def BLIP2_classification_Model(image, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model  = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")

    inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt").to(device)

    # text_features = model.get_text_features(**inputs)
    with torch.no_grad():
        outputs = model(**inputs)

    # this is the image-text similarity score
    logits_per_image = outputs.logits_per_image

    # we can take the softmax to get the label probabilities
    probs = logits_per_image.softmax(dim=1)
    return probs


if __name__ == "__main__":
    image_inp = input("Enter the path to the image: ")
    image = Image.open(image_inp)
    task = input("Choose 1 for classification and 2 for captioning: ")
    if task == "1":
        ALIGN_classification_model(image)
        prompt1 = input("Enter prompt 1: ")
        prompt2 = input("Enter prompt 2: ")
        prompt = [prompt1, prompt2]
    elif task == "2":
        print("Captioning using CLIP")
        # st.write("Image Captioning")
    # model = CLIP_model(image, prompt)