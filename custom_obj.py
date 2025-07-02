from curses.ascii import isalnum
import numpy 
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image, ImageOps
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import re
from transformers import AutoTokenizer
from tqdm import tqdm


class MyTyping:
    sort_dict = lambda x: dict(reversed(sorted(x.items(), key = lambda item: item[1])))


class MyText:
    def download_nltk():
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger_eng')

    def norm_text(text, keep_num = False):
        text = text.lower()
        
        text = text.replace("/", " ")
        text = text.replace("?", "")
        
        if not keep_num:
            text = re.sub(r"\d+", "", text)

        text = " ".join(text.split())
        return text

    class Tokenizer:
        # get <vocab_size> most occur words
        def __init__(self, vocab_size, max_length, tokenizer_name = "dmis-lab/biobert-base-cased-v1.1"):
            self.vocab_size = vocab_size
            self.max_length = max_length
            self.vocab_map = dict({})
            self.sep_vocab = []
            
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            self.cls_id = None
            self.sep_id = None
            self.unknown_id = None
            self.pad_id = 0
            
        def tokenize(self, data):
            tokens = [self.tokenizer.tokenize(s) for s in tqdm(data)]
            return tokens
        
        def to_deep_learning(self, words):
            def get_wordnet_pos(treebank_tag):
                if treebank_tag.startswith('J'):
                    return 'a'
                elif treebank_tag.startswith('V'):
                    return 'v'
                elif treebank_tag.startswith('N'):
                    return 'n'
                elif treebank_tag.startswith('R'):
                    return 'r'
                else:
                    return 'n' 
            
            stop_words = set(stopwords.words("english"))
            white_list = ["what", "when", "where", "why", "any", "how", "if", "more"]
            stop_words.difference_update(white_list)
            
            words = [w for w in words if w not in stop_words]
            pos_tags = nltk.pos_tag(words)
            
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
            
            return words
        
        def preprocess(self, data):
            tokens = self.tokenize(data)
            tokens = [self.to_deep_learning(words) for words in tqdm(tokens)]
            return tokens
            
        def fit(self, tokens):
            word_counts = dict({})
            
            for sentence in tokens:
                for word in sentence:
                    if word not in self.sep_vocab:
                        word_counts[word] = word_counts.get(word, 0) + 1

            word_counts = MyTyping.sort_dict(word_counts)
            vocab = list(word_counts.keys())[:self.vocab_size:]
            self.vocab_size = len(vocab)
            self.vocab_map = {w: i for w, i in zip(vocab, range(1, self.vocab_size + 1))}
            
            self.unknown_id = self.vocab_size + 1
            self.cls_id = self.unknown_id + 1
            self.sep_id = self.cls_id + 1
            
            self.bos_id = self.sep_id + 1  # begin
            self.eos_id = self.bos_id + 1  # end
            
            return word_counts
            
        # pre_ids: extend in the beginning
        # post_ids: extend in the end
        def transform(self, tokens, adaptive_max_length = False, post_ids = [], pre_ids = []):
            ids = []
            
            for sentence in tokens:
                tmp = []
                tmp.extend(pre_ids)
                
                for word in sentence:
                    if word in self.sep_vocab:
                        id = self.sep_id
                    else:
                        id = self.vocab_map.get(word, self.unknown_id)
                    tmp.append(id)
                
                tmp.extend(post_ids)
                ids.append(tmp)
                
            if adaptive_max_length:
                self.max_length = max(list(map(len, ids)))
                
            return ids
            
        def pad_or_trunc(self, ids, max_length = None, padding = "pre", truncation = "post"):
            if max_length is None:
                max_length = self.max_length
            
            for i, sentence in enumerate(ids):
                if len(sentence) > max_length:
                    if truncation == "pre":
                        ids[i] = ids[i][-max_length::]
                    elif truncation == "post":
                        ids[i] = ids[i][:max_length:]
                else:
                    pad_len = max_length - len(sentence)
                    if padding == "pre":
                        ids[i] = [self.pad_id] * pad_len + ids[i]
                    elif padding == "post":
                        ids[i] = ids[i] + [self.pad_id] * pad_len

            return ids
        
        @property
        def all_vocab_size(self):
            return self.vocab_size + 6
            
        @property
        def id_map(self):
            id_map = {v: k for k, v in self.vocab_map.items()}
            
            id_map[self.pad_id] = "<PAD>"
            id_map[self.unknown_id] = "<UKN>"
            id_map[self.cls_id] = "<CLS>"
            id_map[self.sep_id] = "<SEP>"
            id_map[self.bos_id] = "<BOS>"
            id_map[self.eos_id] = "<EOS>"
            
            return id_map
    

class MyImage:
    def change_size(image, target_size, pad_value=0):
        
        target_h, target_w = target_size
        img_h, img_w = image.shape[:2]
        image = cv2.resize(image, (target_w * img_w // img_h, target_h))
        img_h, img_w = image.shape[:2]

        # crop
        start_x = max(0, (img_w - target_w) // 2)
        start_y = max(0, (img_h - target_h) // 2)
        end_x = start_x + min(target_w, img_w)
        end_y = start_y + min(target_h, img_h)

        cropped = image[start_y:end_y, start_x:end_x]

        # pad
        pad_h = max(0, target_h - cropped.shape[0])
        pad_w = max(0, target_w - cropped.shape[1])
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if image.ndim == 3:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))

        padded = np.pad(cropped, padding, mode='constant', constant_values=pad_value)

        return padded