import torch
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

# INSTRUCTION = (
#     "You are a medical vision-language assistant; given an endoscopic image and a clinical "
#     "question that may ask about one or more findings, provide a concise, clinically accurate "
#     "response addressing all parts of the question in natural-sounding medical language as if "
#     "spoken by a doctor in a single sentence."
# )

INSTRUCTION = "You are a medical vision assistant about gastroIntestinal image"


class BaseDataset(Dataset):
    # mode = "train" or "infer" or "both"
    def __init__(
        self, 
        df, 
        processor, 
        mode, 
        img_size, 
        contain_label = True, 
        transform = None
    ):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.img_size = img_size
        
        self.transform = transform
        if self.transform is None:
            self.transform = {}
        self.transform = ImageUtils.get_transform(**self.transform)

        self.data = df.to_dict(orient = 'records')
        self.img_dict = ImageUtils.get_img_dict()
        self.contain_label = contain_label
        
        self.index = None
        self.quest = None
        self.ans = None
        self.img = None
        
    def process(self):
        self.quest = self.data[self.index]["question"].strip()
        self.quest = TextUtils.norm_text(
            self.quest,
            final_char = self.quest[-1] if self.quest[-1] in [".", "?"] else "?"
        )

        if self.contain_label:
            self.ans = self.data[self.index]["answer"].strip()    
            self.ans = TextUtils.norm_text(
                self.ans,
                final_char = ".",
            )
        
        self.img = Image.open(self.img_dict[self.data[self.index]["img_id"]]).convert("RGB")
        self.img = ImageUtils.change_size(self.img, self.img_size)
        if self.mode == "train":
            self.img = self.transform(self.img)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        self.index = index
        return self.process()
    
class CausalDataset(BaseDataset):
    def __init__(
        self, 
        df, 
        processor, 
        mode,  
        img_size, 
        max_full_length,
        max_user_length,
        max_assistant_length,
        contain_label = True,
        transform = None
    ):
        super().__init__(df, processor, mode, img_size, contain_label, transform)
        self.max_full_length = max_full_length
        self.max_user_length = max_user_length
        self.max_assistant_length = max_assistant_length
    
    def process(self):
        super().process()
        inp_mes = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.img,
                    },
                    {
                        "type": "text",
                        "text": self.quest,
                    }
                ]
            }
        ]
        
        out_mes = []
        if self.contain_label:
            out_mes = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": self.ans,
                        }
                    ]
                }
            ]
        
        merge_mes = inp_mes + out_mes
        
        self.img, _ = process_vision_info(merge_mes)
            
        if self.mode == "infer" or self.mode == "both":
            inp_text = self.processor.apply_chat_template(inp_mes, tokenize = False, add_generation_prompt = True)
            inp = self.processor(
                text = inp_text,
                images = self.img,
                padding = "max_length",
                truncation = True,
                max_length = self.max_user_length,
                return_tensors = "pt"
            )
            inp["label"] = self.processor.tokenizer(
                text = self.ans,
                padding = "max_length",
                truncation = True,
                max_length = self.max_assistant_length,
                return_tensors = "pt"
            )
            
            if self.mode == "infer":
                return inp
        
        if self.mode == "train" or self.mode == "both":
            merge_text = self.processor.apply_chat_template(merge_mes, tokenize = False, add_generation_prompt = False)
            
            merge = self.processor(
                text = merge_text,
                images = self.img,
                padding = "max_length",
                truncation = True,
                max_length = self.max_full_length,
                return_tensors = "pt"
            )
            merge = {k: v.squeeze(0) for k, v in merge.items()}
            
            if not self.contain_label:
                return merge
            
            assistant_pattern = self.processor.tokenizer.encode("<|im_start|>assistant\n")
            assistant_idx = find_subsequence(merge["input_ids"].tolist(), assistant_pattern)
            inp_len = assistant_idx + len(assistant_pattern)
            
            label = merge["input_ids"].clone()
            label[:inp_len:] = -100
            label[label == self.processor.tokenizer.pad_token_id] = -100

            merge["labels"] = label
        
            if self.mode == "train":
                return merge

        if self.mode == "both":
            return inp, merge
    
    
# format data for test
class BaseDataFormatter():
    def __init__(self):
        self.batch = None
        self.input = None
        self.output = None
        self.label = None
        self.processor = None
        
    def fn(self):
        self.label[self.label == -100] = self.processor.tokenizer.pad_token_id
    
    def __call__(self, processor, batch, input, output, label):
        self.processor = processor
        self.batch = batch
        self.input = input
        self.output = output
        self.label = label
        
        self.fn()
        
        return self.input, self.output, self.label
    
class CausalDataFormatter(BaseDataFormatter):
    def fn(self):
        super().fn()
        self.output = self.output[::, self.input.shape[-1]::]