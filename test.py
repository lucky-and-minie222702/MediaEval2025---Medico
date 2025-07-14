from custom_obj import *
from my_dataset import *
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

torch.set_float32_matmul_precision("high")

# testing config
batch_size = int(MyCLI.get_arg("batch_size", 16))

# testing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Test on: {device}")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
model.load_state_dict(torch.load('models/model.torch'))
model = model.to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# data
_, _, test_dl = load_data(processor, batch_size = 3)

# test
with torch.no_grad():
    model.eval()
    for i, batch in enumerate(test_dl):
        if i == 1:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        outputs = model(**batch)
        prediction = torch.argmax(outputs.logits, dim = -1)
        
        labels[labels == -100] = 0
        print("Question:", MyText.get_sentence(processor, batch["input_ids"]))
        print("Model:", MyText.get_sentence(processor, prediction))
        print("Actual:", MyText.get_sentence(processor, labels))