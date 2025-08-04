import numpy as np
from PIL import Image, ImageOps
import json
import sys
from transformers.utils import logging
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


class MyDisplay:
    @staticmethod
    def set_trainer_ncols(ncols):
        original_bar = logging.get_progress_bar

        def custom_progress_bar(*args, **kwargs):
            kwargs["ncols"] = ncols
            return original_bar(*args, **kwargs)

        logging.get_progress_bar = custom_progress_bar
        


class MyConfig:
    @staticmethod
    def load_json(p):
        data = dict({})
        with open(p, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data


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

        # rouge
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

        # meteor
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