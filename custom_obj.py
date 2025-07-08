import numpy 
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


class MyTyping:
    sort_dict = lambda x: dict(reversed(sorted(x.items(), key = lambda item: item[1])))


def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download('averaged_perceptron_tagger_eng')


class MyText:
    class MyTokenizer:
        # get <vocab_size> most occur words
        def __init__(self, vocab_size, max_length):
            self.vocab_size = vocab_size
            self.max_length = max_length
            self.vocab_map = dict({})
            
            self.unknown_id = None
            self.pad_id = 0
            
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.stem_to_orignal = dict({})
            
            self.stop_words = set(stopwords.words("english"))
            stop_words_white_list = ["what", "when", "where", "why", "any", "how", "if", "more"]
            self.stop_words.difference_update(stop_words_white_list)
            
        def norm_text_(self, text, keep_num = True):
                text = text.lower()
                
                text = text.replace("/", " ")
                text = text.replace("?", "")
                text = text.replace(".", ",")
                text = text.replace(";", ",")
                
                if not text[-1].isalnum():
                    text = text[:-1:]
                
                if not keep_num:
                    text = re.sub(r"\d+", "", text)
                    
                text = re.sub(r'([^a-zA-Z0-9])', r' \1 ', text)

                text = " ".join(text.split())
                return text
        
        def remove_stopwords_(self, words):
            out = [w for w in words if w not in self.stop_words]
            return out
        
        def to_deep_learning_(self, words):
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

            out = [w.strip() for w in words]

            pos_tags = nltk.pos_tag(out)
            
            out = [self.lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
            for i, word in enumerate(out):
                original = out [i]
                stem = self.stemmer.stem(word)
                out[i] = stem
                self.stem_to_orignal[stem] = original

            return out
            
        def norm_text(self, data, keep_num = True):
            out = [self.norm_text_(t, keep_num = keep_num) for t in tqdm(data, desc = "Normalize text")]
            return out
            
        def tokenize(self, data):
            tokens = [word_tokenize(s) for s in tqdm(data, desc = "Tokenize")]
            return tokens
        
        def remove_stopwords(self, data):
            out = [self.remove_stopwords_(t) for t in tqdm(data, desc = "Remove stop words")]
            return out
        
        def to_deep_learning(self, data):
            out = [self.to_deep_learning_(t) for t in tqdm(data, desc = "To deep learning format")]
            return out
        
        def preprocess(self, data, remove_stopwords = True, to_deep_learning = True):
            norm_data = self.norm_text(data)
            tokens = self.tokenize(norm_data)
            
            if remove_stopwords:
                tokens = self.remove_stopwords(tokens)

            if to_deep_learning:
                tokens = self.to_deep_learning(tokens)

            return tokens
            
        def fit(self, tokens):
            word_counts = dict({})
            
            for sentence in tokens:
                for word in sentence:
                    word_counts[word] = word_counts.get(word, 0) + 1

            word_counts = MyTyping.sort_dict(word_counts)
            vocab = list(word_counts.keys())[:self.vocab_size:]
            self.vocab_size = len(vocab)
            self.vocab_map = {w: i for w, i in zip(vocab, range(1, self.vocab_size + 1))}
            
            self.unknown_id = self.vocab_size + 1
            
            return word_counts
            
        # pre_ids: extend in the beginning
        # post_ids: extend in the end
        def transform(self, tokens, adaptive_max_length = False, post_ids = [], pre_ids = []):
            ids = []
            
            for sentence in tokens:
                tmp = []
                tmp.extend(pre_ids)
                
                for word in sentence:
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
        
        def get_id(self, vocab):
            return self.vocab_map.get(vocab, None)
        
        def get_vocab(self, id):
            return self.id_map.get(id, None)
        
        def add_vocab(self, vocab):
            if vocab not in self.vocab_map:
                self.vocab_map[vocab] = self.vocab_size + 1
                self.vocab_size += 1
                self.unknown_id += 1
                return True
            return False
        
        def decode_sentence_(self, ids):
            out = []
            for i in ids:
                if i != self.pad_id:
                    vocab = self.get_vocab(i)
                    vocab = self.stem_to_orignal.get(vocab, vocab)
                    out.append(vocab)
            return out
        
        def decode_sentence(self, data):
            out = [self.decode_sentence_(s) for s in data]
            return out
        
        @property
        def all_vocab_size(self):
            return self.vocab_size + 2
            
        @property
        def id_map(self):
            id_map = {v: k for k, v in self.vocab_map.items()}
            
            id_map[self.pad_id] = "<PAD>"
            id_map[self.unknown_id] = "<UKN>"
            
            return id_map
        
    def bleu_score(reference, candidate, smooth = True):
        smoothie = SmoothingFunction().method4
        score = sentence_bleu([reference], candidate, smoothing_function = smoothie if smooth else None)
        return score
    
    def bleu_score_batch(reference, candidate, smooth = True):
        scores = [MyText.bleu_score(r, c, smooth) for r, c in zip(reference, candidate)]
        return np.mean(scores)
    

class MyImage:
    def change_size(img: ImageFile, target_size, fill_color = (0, 0, 0)):
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