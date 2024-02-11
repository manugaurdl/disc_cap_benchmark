import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
from utils.utils import *
import streamlit as st

def ALIGN_classification_model(image, prompt):
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
    model = AlignModel.from_pretrained("kakaobrain/align-base")

    inputs = processor(text=prompt, images=image, return_tensors="pt")

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