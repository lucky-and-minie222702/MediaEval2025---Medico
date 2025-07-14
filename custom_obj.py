import joblib
import numpy as np
from PIL import Image, ImageOps, ImageFile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import json
import sys
import evaluate


class MyUtils:
    @staticmethod
    def get_scores_from_ids(processor, pred, label, exclude_metrics = []):
        pred = pred.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        
        pred = processor.tokenizer.batch_decode(pred, skip_special_tokens = True)
        label = processor.tokenizer.batch_decode(label, skip_special_tokens = True)    
        
        return MyText.get_scores(pred, label, exclude_metrics)
    
    @staticmethod
    def get_sentences_from_ids(processor, s):
        s = s.detach().cpu().numpy().tolist()
        s = processor.tokenizer.batch_decode(s, skip_special_tokens = True)    
        return s
    
    @staticmethod
    def load_json(p):
        with open(p, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    class MetricLogger:
        def __init__(self, processor, exclude_metrics = []):
            self.cur_content = None
            self.processor = processor
            self.content = None
            self.exclude_metrics = exclude_metrics
        
        def log_per_step(self, predictions, labels):
            scores = MyUtils.get_scores_from_ids(self.processor, predictions, labels, self.exclude_metrics)
            
            if self.cur_content is None:
                self.cur_content = scores
                for k, v in self.cur_content.items():
                    self.cur_content[k] = [v]
            else:
                for k, v in scores.items():
                    self.cur_content[k].append(v)
                    
        @property
        def mean_content(self):
            return {k: np.mean(v) for k, v in self.cur_content.items()}
                    
        def end_batch(self):
            if self.content is None:
                self.content = self.mean_content
                for k, v in self.content.items():
                    self.content[k] = [v]
            else:
                for k in self.content.keys():
                    self.content[k].append(self.mean_content[k])

            self.cur_content = None
            

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
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    
    @staticmethod
    def get_scores(predictions, references, exclude_metrics = []):
        clean_data = [
            (pred.strip(), ref.strip())
            for pred, ref in zip(predictions, references)
            if pred.strip() and ref.strip()
        ]

        if len(clean_data) == 0:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "meteor": 0.0,
            }

        clean_preds, clean_refs = zip(*clean_data)

        bleu_refs = [[ref] for ref in clean_refs]

        if "bleu" in exclude_metrics:
            bleu = dict({})
        else:
            bleu = MyText.bleu.compute(predictions = clean_preds, references = bleu_refs)
            
        if "rouge" in exclude_metrics:
            rouge = dict({})
        else:
            rouge = MyText.rouge.compute(predictions = clean_preds, references = clean_refs)
            
        if "meteor" in exclude_metrics:
            meteor = dict({})
        else:
            meteor = MyText.meteor.compute(predictions = clean_preds, references = clean_refs)

        return {
            "bleu": bleu.get("bleu", 0.0),
            "rouge1": rouge.get("rouge1", 0.0),
            "rouge2": rouge.get("rouge2", 0.0),
            "rougeL": rouge.get("rougeL", 0.0),
            "meteor": meteor.get("meteor", 0.0),
        }
    

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