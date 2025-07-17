import torch
from torch import nn
from custom_obj import *
from transformers import BlipForConditionalGeneration

MODEL_NAME = {
    "vqa": "Salesforce/blip-vqa-base",
    "caption": "Salesforce/blip-image-captioning-base",
}

