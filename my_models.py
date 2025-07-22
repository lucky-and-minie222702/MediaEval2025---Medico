import torch
from torch import nn
from my_tools import *
from transformers import BlipForConditionalGeneration, BlipProcessor


def get_baseline():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    return model, processor

def get_baseline():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    return model, processor


def get_models_by_name(name):
    if name == "baseline":
        return get_baseline()