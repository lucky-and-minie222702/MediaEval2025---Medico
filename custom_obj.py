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
    def get_scores_from_ids(processor, pred, label):
        pred = pred.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        
        pred = processor.tokenizer.batch_decode(pred, skip_special_tokens = True)
        label = processor.tokenizer.batch_decode(label, skip_special_tokens = True)    
        
        return MyText.get_scores(pred, label)
    
    class MetricLogger:
        def __init__(self, processor):
            self.cur_content = None
            self.processor = processor
            self.content = None
        
        def log_per_step(self, predictions, labels):
            scores = MyUtils.get_scores_from_ids(self.processor, predictions, labels)
            
            if self.cur_content is None:
                self.cur_content = scores
                for k, v in scores.items():
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
    def get_scores(predictions, references):
        references = [[tokens] for tokens in references] 

        bleu = MyText.bleu.compute(predictions = predictions, references=references)

        rouge = MyText.rouge.compute(predictions = predictions, references = [ref[0] for ref in references])

        meteor = MyText.meteor.compute(predictions = predictions, references = [ref[0] for ref in references])

        results = {
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "meteor": meteor["meteor"],
        }

        return results
    

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