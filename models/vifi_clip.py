import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import datetime
import shutil
from pathlib import Path
import time
import numpy as np
import random
import yaml
import argparse
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# import sys
# sys.path.insert(0, "/home2/manugaur/vlm_models/")
from model_repos.vificlip.datasets.blending import CutmixMixupBlending

from model_repos.vificlip.utils.config import get_config
# from utils.optimizer import build_optimizer, build_scheduler
from model_repos.vificlip.utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
# from datasets.build import build_dataloader
from model_repos.vificlip.utils.logger import create_logger
from model_repos.vificlip.utils.config import get_config
from model_repos.vificlip.trainers import vificlip

import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import datetime
import shutil
from pathlib import Path
import time
import numpy as np
import random
import yaml

# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import sys
sys.path.insert(0, "/home2/manugaur/vlm_models/")

import torch
from decord import VideoReader, cpu
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import XCLIPProcessor, XCLIPModel

file_path = "/home2/manugaur/vlm_models/dog.mp4"


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))
# sample 16 frames
vr.seek(0)
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
video = vr.get_batch(indices).asnumpy()
video.shape


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default= "/home2/manugaur/Models-Gallery/model_repos/vificlip/configs/zero_shot/eval/k600/16_32_K600_ZS_split1.yaml")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config



def VIFICLIP_classification_model(video_input, prompt):

    vr = VideoReader(video_input, num_threads=1, ctx=cpu(0))
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
    video = vr.get_batch(indices).asnumpy()

    args, config = parse_option()
    model = vificlip.returnCLIP(config,class_names=prompt)

    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    # model = XCLIPModel.from_pretrained(model_name)
    inputs = processor(text=prompt, videos=list(video), return_tensors="pt", padding=True)
    with torch.no_grad():
        x = inputs['pixel_values']
        x = x.half()

        outputs = model(x.float()).softmax(dim = 1)

    return outputs
if __name__ == '__main__':
    # prepare config
    # vid_input = input("Enter the path to the video: ")
    # prompt1 = input("Enter prompt 1: ")
    # prompt2 = input("Enter prompt 2: ")
    vid_input = "/home2/manugaur/vlm_models/dog.mp4"
    prompt1 = 'cat'
    prompt2 = 'dog'
    prompt3 = 'puppy'
    prompt = [prompt1, prompt2,prompt3]
        
    print(VIFICLIP_classification_model(vid_input, prompt))
    