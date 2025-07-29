import torch
from torch import nn
from my_tools import *
from transformers import AutoProcessor, AutoModelForCausalLM


def get_baseline():
    name = "microsoft/git-base-vqav2"
    processor = AutoProcessor.from_pretrained(name)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, processor

def get_models_by_name(name):
    if name == "baseline":
        return get_baseline()