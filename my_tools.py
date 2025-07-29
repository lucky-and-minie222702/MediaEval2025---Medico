import joblib
import numpy as np
from PIL import Image, ImageOps, ImageFile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import json
import sys
import evaluate
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score



class MyConfig:
    @staticmethod
    def load_json(p):
        with open(p, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    def __init__(self, data, is_path = True):
        self.data= data
        if is_path:
            self.data = MyConfig.load_json(data)
        
    def __getitem__(self, index):
        out = self.data[index]
        return out


class MyUtils:    
    @staticmethod
    def get_sentences_from_ids(processor, s):
        s = s.detach().cpu().numpy().tolist()
        s = processor.tokenizer.batch_decode(s, skip_special_tokens = True)    
        return s

    @staticmethod
    def get_scores_from_ids(processor, pred, label):
        pred = MyUtils.get_sentences_from_ids(processor, pred)
        label = MyUtils.get_sentences_from_ids(processor, label)
        
        return MyText.get_scores(pred, label)
    
    class MetricLogger:
        def __init__(self, processor):
            self.cur_content = None
            self.processor = processor
            self.batch_content = None
            self.step_content = None
        
        def log_per_step(self, predictions, labels):
            scores = MyUtils.get_scores_from_ids(self.processor, predictions, labels)
            
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
        
        @property
        def last_content(self):
            return {k: v[-1] for k, v in self.cur_content.items()}
                    
        def end_batch(self):
            if self.batch_content is None:
                self.batch_content = self.mean_content
                for k, v in self.batch_content.items():
                    self.batch_content[k] = [v]
            else:
                for k in self.batch_content.keys():
                    self.batch_content[k].append(self.mean_content[k])

            if self.step_content is None:
                self.step_content = self.cur_content
            else:
                for k, v in self.cur_content:
                    self.step_content[k].extend(v)

            self.cur_content = None
            
        @property
        def content(self):
            return {
                "per_step": self.step_content,
                "per_batch": self.batch_content,
            }
            
    class TestLogger(MetricLogger):
        def __init__(self, processor):
            super().__init__(processor)
            self.outputs = None
            self.losses = []
            
        def log_per_step(self, questions, predictions, labels, loss):
            super().log_per_step(predictions, labels)
            
            cur_outputs = [
                MyUtils.get_sentences_from_ids(self.processor, questions),
                MyUtils.get_sentences_from_ids(self.processor, labels),
                MyUtils.get_sentences_from_ids(self.processor, predictions),
            ]
            
            if self.outputs is None:
                self.outputs = cur_outputs
            else:
                self.outputs = np.concatenate([self.outputs, cur_outputs], axis = 1)  # (3, n_samples)
            
            self.losses.append(loss)
            
        def end_batch(self):
            super().end_batch()
            
            self.outputs = np.transpose(self.outputs, (1, 0)) # (n_samples, 3)
            
        @property
        def content(self):
            return {
                "per_step": self.step_content,
                "per_batch": self.batch_content,
                "losses": self.losses,
                "outputs": self.outputs,
            }
    

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


class MyText:
    # bleu = evaluate.load("bleu")
    # rouge = evaluate.load("rouge")
    # meteor = evaluate.load("meteor")
    
    bleu = corpus_bleu
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = True)
    meteor = meteor_score
    
    @staticmethod
    def get_scores(predictions, references):
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

        clean_refs_list = [[ref] for ref in clean_refs]

        bleu = corpus_bleu(clean_preds, clean_refs_list).score / 100

        # --- ROUGE (F1 averaged)
        r1_total, r2_total, rl_total = 0, 0, 0
        for pred, refs in zip(clean_preds, clean_refs_list):
            ref = refs[0]
            scores = MyText.rouge.score(ref, pred)
            r1_total += scores["rouge1"].fmeasure
            r2_total += scores["rouge2"].fmeasure
            rl_total += scores["rougeL"].fmeasure
        n = len(clean_preds)
        rouge1 = r1_total / n
        rouge2 = r2_total / n
        rougeL = rl_total / n

        # --- METEOR (averaged)
        meteor_total = 0
        for pred, refs in zip(clean_preds, clean_refs_list):
            meteor_total += meteor_score(
                [ref.split() for ref in refs],
                pred.split()
            )
        meteor = meteor_total / n

        return {
            "bleu": bleu,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "meteor": meteor
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