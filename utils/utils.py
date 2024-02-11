# import sys
# sys.path.insert(0, "/home2/manugaur/vlm_models/ViFi-CLIP")
import streamlit as st
import gc
from PIL import Image
from models.clip import *
from models.blip import *
from models.align import *
from models.clip4clip import *
from models.xclip import *
from models.vifi_clip import *
from models.blip2 import *



def clear_ada_cache(model, processor):
    """
    Function to clear cache in ada on task change in different models
    """

    print("Clearing cache in ada")
    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()
    # del model
    # gc.disable()
    # return model, processor
    
#------------------------------------------------------------------------------------------------------------------------
                                            #IMAGE HELPER    

def choose_task(uploaded_file, model_type):
    """
    Function to choose the task
    """
    task = st.selectbox("Select the task",
                        ("None", "Image Classification", "Image Captioning", "Question Answering"), key=model_type+"_task")
    if task == "Image Classification":
        # print("Classification using "+model_type)
        get_classification_prompts(uploaded_file, model_type)
    elif task == "Image Captioning":
        print("Captioning using "+model_type)
        st.write("Image Captioning")
    elif task== "Question Answering":
        print("Question Answering using "+model_type)
        get_answer(uploaded_file, model_type)


def get_answer(uploaded_file, model_type):
    """
    Function to get the answer
    """
    question = st.text_input(
        'Enter question', value="What is the name of the ", key=model_type+"question")
    image = Image.open(uploaded_file)
        
    if (question != "" and question != "What is the name of the "):
        answer = question_answering_models(image, question, model_type)
        st.write("Answer: ", answer)
    else:
        st.markdown(":red[Please enter the question]")

def question_answering_models(image, question, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache(model, processor)

    if model_type=="BLIPv2":
        print("Loading BLIPv2 classification model")
        return BLIP2_Question_Answering_Model(image, question)



def get_classification_prompts(uploaded_file, model_type):
    """
    Function to get the classification prompts
    """

    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key=model_type+"prompt1")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key=model_type+"prompt2")
    prompt = [prompt1, prompt2]
    
    image = Image.open(uploaded_file)
        
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        probs = classification_models(image, prompt, model_type)
        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")

def classification_models(image, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache(model, processor)

    if model_type=="ALIGN":
        print("Loading ALIGN classification model")
        return ALIGN_classification_model(image, prompt)
    
    elif model_type=="CLIP":
        print ("Loading CLIP classification model")
        st.spinner("Loading CLIP classification model")
        return CLIP_classification_model(image, prompt)
    
    elif model_type=="BLIP":
        print ("Loading BLIP classification model")
        return BLIP_classification_model(image, prompt)
    
    elif model_type=="BLIPv2":
        print ("Loading BLIPv2 classification model")
        return BLIP2_classification_Model(image, prompt)
    
#------------------------------------------------------------------------------------------------------------------------
                                            #VIDEO HELPERS

def choose_vid_task(uploaded_file, model_type):

    task = st.selectbox("Select the task",
                        ("None", "Video Classification", "Video Captioning"), key=model_type+"_task")
    if task == "Video Classification":
        # print("Classification using "+model_type)
        get_vid_cls_prompts(uploaded_file, model_type)
    elif task == "Image Captioning":
        print("Captioning using "+model_type)
        st.write("Image Captioning")        

def get_vid_cls_prompts(uploaded_file, model_type):

    prompt1 = st.text_input(
        'Enter prompt 1', value="A video of a ", key=model_type+"prompt1")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A video of a ", key=model_type+"prompt2")
    prompt = [prompt1, prompt2]

    #####reading video inpt
    # image = Image.open(uploaded_file)
        
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A video of a " and prompt2 != "A video of a "):
        probs = vid_cls_models(uploaded_file, prompt, model_type)
        st.write("Probability of prompt 1: ","{:.2f}".format(probs.detach().numpy()[0][0]))
        st.write("Probability of prompt 2: ","{:.2f}".format(probs.detach().numpy()[0][1]))        
    else:
        st.markdown(":red[Please enter both the prompts]")



def vid_cls_models(video, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache(model, processor)

    if model_type=="CLIP4CLIP":
        print("Loading CLIP4CLIP classification model")
        st.spinner("Loading CLIP4CLIP classification model")
        return CLIP4CLIP_classification_model(video, prompt)
    
    elif model_type=="XCLIP":
        print ("Loading XCLIP classification model")
        st.spinner("Loading XCLIP classification model")
        return XCLIP_classification_model(video, prompt)
    elif model_type=="VIFICLIP":
        print ("Loading VIFI-CLIP classification model")
        st.spinner("Loading VIFI-CLIP classification model")
        return VIFICLIP_classification_model(video, prompt)
    
    elif model_type=="BLIP":
        print ("Loading BLIP classification model")
        return BLIP_classification_model(video, prompt)
    
