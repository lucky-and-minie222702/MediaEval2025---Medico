import joblib
from my_tools import *
from my_dataset import *
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_models import *

torch.set_float32_matmul_precision("high")

# load config
config = MyConfig("test_config.json")

# training config
batch_size = config["batch_size"]

# models and training strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")
model, processor = get_models_by_name(config["model"])

# data
test_dl = load_data(processor, max_question_length = config["dataset"]["mql"], max_answer_length = config["dataset"]["mal"], batch_size = batch_size, use_original = config["dataset"]["use_original"], test_only = True)

# logger
logger = MyUtils.TestLogger(processor)

# save path
folder = f"models_checkpoint_{config['name']}/"
model.load_state_dict(torch.load(folder + "model.torch"))
model = model.to(device)

with torch.no_grad():
    model.eval()
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        loss = outputs.loss
        
        predictions = torch.argmax(outputs.logits, dim = -1)
            
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
            
        questions = batch["input_ids"]

        logger.log_per_step(questions, predictions, labels, loss.item())
    
logger.end_batch()

for k, v in logger.content.items():
    print(f"{k}: {v}")

joblib.dump(logger.content, folder + "test_metrics.joblib")
joblib.dump(logger.outputs, folder + "test_outputs.joblib")