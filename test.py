import joblib
from custom_obj import *
from my_dataset import *
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_float32_matmul_precision("high")


batch_size = int(MyCLI.get_arg("batch_size", 16))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

model.load_state_dict(torch.load('models/model.torch'))

_, _, test_dl = load_data(processor, batch_size = 1)


def get_sentence(s):
    s = s.detach().cpu().numpy().tolist()
    s = processor.tokenizer.batch_decode(s, skip_special_tokens = True)    
    return s


with torch.no_grad():
    for i, batch in enumerate(test_dl):
        if i == 1:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        print("Question:", get_sentence(batch["input_ids"]))
        print("Answer:", get_sentence(batch["labels"]))