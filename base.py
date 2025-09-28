import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


class ModelInterface:
    def __init__(
        self, 
        pretrained_name, 
        model_class = None, 
        processor_class = None
    ):  
        self.pretrained_name = pretrained_name
        self.model_class = model_class
        self.processor_class = processor_class
        
        if self.model_class is None:
            self.model_class = AutoModel
        if self.processor_class is None:
            self.processor_class = AutoProcessor

        self.model = model_class.from_pretrained(
            pretrained_name,
            dtype = torch.bfloat16,
            device_map = "auto",
            trust_remote_code = True
        )
        self.model.enable_input_require_grads()

        self.processor = processor_class.from_pretrained(self.pretrained_name)
        
    def to_lora(self, **kwargs):
        lora_config = LoraConfig(**kwargs)
        
        self.model = get_peft_model(self.model, lora_config)
        
    def infer(self, dl, returns =  ["output"], disable_tqdm = True, generation_config = {}):
        inputs = []
        outputs = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(dl, disable = disable_tqdm):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                output = self.model.generate(
                    **batch,
                    **generation_config
                )
                
                if "input" in returns:
                    inputs.append(batch["input_ids"])
                
                if "output" in returns:
                    outputs.append(output)
                    
                    
                if "label" in batch:
                    if "label" in returns:
                        labels.append(batch["labels"])
                    
                label = batch["labels"]   
                label[label == -100] == self.processor.tokenizer.pad_token_id
        
        return inputs, outputs, labels
            
    def test(
        self, 
        dl,
        output_dir = None, 
        generation_config = None, 
        format_data_fn = None,
    ):
        if generation_config is None:
            generation_config = {
                "do_sample": False,
            }

        logger = ModelUtils.TestLogger(self.processor)
        
        with torch.no_grad():
            pbar = tqdm(dl)
            for batch in pbar:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                loss = self.model(**batch).loss.item()
                
                input = batch["input_ids"]
                
                label = batch["labels"]   
                label[label == -100] == self.processor.tokenizer.pad_token_id
                
                output = self.model.generate(
                    **batch,
                    **generation_config
                )
                
                if format_data_fn is not None:
                    input, output, label = format_data_fn(self.processor, batch, input, output, label)

                logger.log_per_step(
                    quest = input,
                    pred = output,
                    label = label,
                    loss = loss,
                    n_returns = generation_config.get("num_return_sequences", 1),
                )
                pbar.set_postfix(loss = round(np.mean(logger.loss), 3), **{k: round(v, 3) for k, v in logger.cur_scores.items()})
        
        logger.end()

        if output_dir is not None:
            joblib.dump(logger.results, f"{output_dir}/test.results")
            
        print(f"loss: {logger.loss:.4f}")
        for k, v in logger.scores.items():
            print(f"{k}: {v:.4f}")

        return logger


INSTRUCTION = "You are a medical vision assistant about gastrointestinal images."

# dataset	     
class BaseDataset(Dataset):
    def __init__(self, df, processor, mode, img_size, transform = None):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.img_size = img_size
        
        self.transform = transform
        if self.transform is None:
            self.transform = {}
        self.transform = ImageUtils.get_transform(**transform)

        self.data = df.to_dict(orient = 'records')
        self.img_dict = ImageUtils.get_img_dict()
        
    def process(self, index):
        pass
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.process(index)
    
class CausalDataset(BaseDataset):
    def __init__(
        self, 
        df, 
        processor, 
        mode,  # "train" or "infer"
        img_size, 
        max_length,
        is_qwen = False,
        transform = None
    ):
        super().__init__(df, processor, mode, img_size, transform)
        self.max_length = max_length
        self.is_qwen = is_qwen
    
    def process(self, index):
        quest = self.data[index]["question"].strip()
        ans = self.data[index]["answer"].strip()
        
        quest = TextUtils.norm_text(
            quest,
            final_char = quest[-1] if quest[-1] in [".", "?"] else "?"
        )
        
        ans = TextUtils.norm_text(
            ans,
            final_char = ".",
        )
        
        img = Image.open(self.img_dict[self.data[index]["img_id"]]).convert("RGB")
        img = ImageUtils.change_size(img, self.img_size)
        
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
                        "image": img,
                    },
                    {
                        "type": "text",
                        "text": quest,
                    }
                ]
            }
        ]
        
        out_mes = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": ans,
                    }
                ]
            }
        ]
        
        merge_mes = inp_mes + out_mes
        
        if self.is_qwen:
            img, _ = process_vision_info(inp_mes)
        
        inp_text = self.processor.apply_chat_template(inp_mes, tokenize = False, add_generation_prompt = self.mode == "infer")
        merge_text = self.processor.apply_chat_template(merge_mes, tokenize = False, add_generation_prompt = False)
        
        inp = self.processor(
            text = inp_text,
            images = img,
            padding = False,
            truncation = False,
            return_tensors = "pt"
        )
        
        merge = self.processor.tokenizer(
            text = merge_text,
            padding = False,
            truncation = False,
            return_tensors = "pt"
        )

        inp = {k: v.squeeze(0) for k, v in inp.items()}
        merge = {k: v.squeeze(0) for k, v in merge.items()}
        
        inp_len = inp["input_ids"].shape[0]
        
        if self.mode == "train":
            label = merge["input_ids"].clone()
            label[merge["attention_mask"]] = -100
            label[:inp_len:] = -100

            merge["labels"] = label
    
            merge["input_ids"] = ModelUtils.pad_and_trunc(merge["input_ids"], self.max_length, self.processor.tokenizer.pad_token_id)
            merge["attention_mask"] = ModelUtils.pad_and_trunc(merge["attention_mask"], self.max_length, 0)
            merge["labels"] = ModelUtils.pad_and_trunc(merge["labels"], self.max_length, -100)
        elif self.mode == "infer":
            merge["labels"] = merge["input_ids"].clone()[inp_len::]
            merge["input_ids"] = ModelUtils.pad_and_trunc(merge["input_ids"], self.max_length, self.processor.tokenizer.pad_token_id)
            merge["attention_mask"] = ModelUtils.pad_and_trunc(merge["attention_mask"], self.max_length, 0)
            merge["labels"] = ModelUtils.pad_and_trunc(merge["labels"], self.max_length, -100)
            
        return merge
    
    
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
        self.output = self.output[::, :self.input.shape[-1]:]
    
    def __call__(self, processor, batch, input, output, label):
        self.processor = processor
        self.batch = batch
        self.input = input
        self.output = output
        self.label = label
        
        self.fn()
        
        return self.input, self.output, self.label