import torch
from torch import nn
from my_tools import *
from transformers import BlipForConditionalGeneration, BlipProcessor


def get_baseline():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    return model, processor