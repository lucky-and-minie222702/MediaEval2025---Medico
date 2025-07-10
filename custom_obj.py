import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image, ImageOps, ImageFile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import json
import sys


class MyCLI:
    @staticmethod
    def get_arg(name, default = None):
        if name in sys.argv:
            i = sys.argv.index(name)
            try:
                return sys.argv[i+1]
            except:
                return default
        return default


class MyTyping:
    sort_dict = lambda x: dict(reversed(sorted(x.items(), key = lambda item: item[1])))
    reformat = lambda s: json.loads(s.replace("\n", ""))


def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download('averaged_perceptron_tagger_eng')


class MyText:
    @staticmethod
    def bleu_score(reference, candidate, smooth = True):
        smoothie = SmoothingFunction().method4
        score = sentence_bleu([reference], candidate, smoothing_function = smoothie if smooth else None)
        return score
    
    @staticmethod
    def bleu_score_batch(reference, candidate, smooth = True):
        scores = [
            MyText.bleu_score(
                r, 
                c, 
                smooth,
            ) for r, c in zip(reference, candidate)]
        return np.mean(scores)
    

class MyImage:
    @staticmethod
    def change_size(img, target_size, fill_color = (0, 0, 0)):
        target_h, target_w = target_size
        img_h, img_w = img.size
        img = img.resize((target_w * img_w // img_h, target_h), resample = Image.BICUBIC)
        img_h, img_w = img.size

        w, h = img.size
        target_w, target_h = target_size

        if w > target_w or h > target_h:
            return ImageOps.fit(img, target_size, method = Image.BICUBIC, centering = (0.5, 0.5))
        else:
            return ImageOps.pad(img, target_size, method = Image.BICUBIC, color = fill_color, centering = (0.5, 0.5))