import joblib
from custom_obj import *
from my_dataset import *
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

torch.set_float32_matmul_precision("high")


batch_size = int(MyCLI.get_arg("batch_size", 16))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Test on: {device}")
model = torch.load('models/model.torch', map_location = torch.device('cpu') if not torch.cuda.is_available() else None)
model = model.to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

_, _, test_dl = load_data(processor, batch_size = 3)


def get_sentence(s):
    s = s.detach().cpu().numpy().tolist()
    s = processor.tokenizer.batch_decode(s, skip_special_tokens = True)    
    return s


with torch.no_grad():
    model.eval()
    for i, batch in enumerate(test_dl):
        if i == 1:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        outputs = model.generate(**batch, max_length = 40)
        
        labels[labels == -100] = 0
        print("Question:", get_sentence(batch["input_ids"]))
        print("Model:", get_sentence(outputs))
        print("Actual:", get_sentence(labels))