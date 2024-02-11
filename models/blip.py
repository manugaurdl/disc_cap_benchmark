from PIL import Image
from transformers import AutoProcessor, BlipModel
from utils.utils import *
import streamlit as st


def BLIP_classification_model(image, prompt):
    # print("hello")
    # print(prompt)
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs


if __name__ == "__main__":
    image_inp = input("Enter the path to the image: ")
    image = Image.open(image_inp)
    prompt1 = input("Enter prompt 1: ")
    prompt2 = input("Enter prompt 2: ")
    prompt = [prompt1, prompt2]
    model = BLIP_model(image, prompt)