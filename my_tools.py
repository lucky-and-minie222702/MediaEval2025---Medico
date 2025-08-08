from PIL import Image, ImageOps
import json
import sys
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
import torch
import os
import numpy as np


class MyConfig:
    @staticmethod
    def load_json(p):
        data = dict({})
        with open(p, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data


class MyUtils: 
    @staticmethod
    def get_sentences_from_ids(processor, s, to_numpy = False):
        s = s.detach().cpu().numpy().tolist()
        s = processor.tokenizer.batch_decode(s, skip_special_tokens = True)    
        if to_numpy:
            s = np.array(s)
        return s

    @staticmethod
    def get_scores_from_ids(processor, pred, label):
        pred = MyUtils.get_sentences_from_ids(processor, pred)
        label = MyUtils.get_sentences_from_ids(processor, label)
        
        return MyText.get_scores(pred, label)
    
    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle = True):
        def collate_fn(batch):
            return {
                key: torch.stack([item[key] for item in batch])
                for key in batch[0]
            }

        return DataLoader(
            dataset = dataset, 
            batch_size = batch_size, 
            shuffle = shuffle, 
            num_workers = 4, 
            persistent_workers = True, 
            pin_memory = False, 
            collate_fn = collate_fn
        )
        
    @staticmethod
    def get_latest_checkpoint(p):
        checkpoints = [d for d in os.listdir(f"results/{p}") if d.startswith("checkpoint")]
        
        if not checkpoints:
            return None

        checkpoints = list(map(lambda x: int(x.split("-")[1]), checkpoints))
        latest = sorted(checkpoints)[-1]
        return latest
        
    class TestLogger():
        def __init__(self, processor):
            self.processor = processor
            self.scores = {
                "bleu": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "meteor": [],
            }
            self.outputs = {
                "questions": [],
                "predictions": [],
                "labels": [],
            }
            
        def log_per_step(self, quest, pred, label, n_returns):
            quest = MyUtils.get_sentences_from_ids(self.processor, quest, to_numpy = True)
            pred = MyUtils.get_sentences_from_ids(self.processor, pred, to_numpy = True).reshape(-1, n_returns)
            label = MyUtils.get_sentences_from_ids(self.processor, label, to_numpy = True)
            
            self.outputs["questions"].append(quest)
            self.outputs["predictions"].append(pred)
            self.outputs["labels"].append(label)
            
            all_scores = {
                "bleu": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "meteor": [],
            }
            
            for i in range(n_returns):
                scores = MyText.get_scores(pred[::, i], label)
                for k, v in scores.items():
                    all_scores[k].append(v)
            
            for k, v in all_scores.items():
                self.scores[k].append(v)
                
        def end(self):
            for k, v in self.outputs.items():
                self.outputs[k] = np.concatenate(v, axis = 0)
                
            for k, v in self.scores.items():
                self.scores[k] = np.mean(v, axis = 0)
                
        @property
        def cur_scores(self):
            cur_scores = self.scores.copy()
            for k, v in cur_scores.items():
                cur_scores[k] = np.max(
                    np.mean(v, axis = 0), axis = -1
                )
            return cur_scores
                
        @property
        def results(self):
            return {
                "outputs": self.outputs,
                "scores": self.scores,
            }


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