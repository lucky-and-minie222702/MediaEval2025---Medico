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

# testing config
batch_size = config["batch_size"]
use_tqdm = config["use_tqdm"]

# models and testing strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Test on: {device}")
model, processor = get_models_by_name(config["model"])

# data
test_dl = load_data(processor, max_question_length = config["dataset"]["mql"], max_answer_length = config["dataset"]["mal"], train_ratio = config["train_ratio"], batch_size = batch_size, use_original = config["dataset"]["use_original"], complexities = config["dataset"]["complexities"], complexity_weight = config["dataset"]["complexity_weight"], test_only = True)

# logger
logger = MyUtils.TestLogger(processor)

# save path
folder = f"models_checkpoint_{config['name']}/"
model.load_state_dict(torch.load(folder + "model.torch", map_location = device))
model = model.to(device)

tqdm_wrapper = lambda dl, name: tqdm(dl, desc = f"{name}", ncols = 175, disable = not use_tqdm)

with torch.no_grad():
    model.eval()
    pbar = tqdm_wrapper(test_dl, "Test ")
    for batch, _ in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        outputs = model(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            labels = labels,
        )
        
        loss = outputs.loss
        
        predictions = torch.argmax(outputs.logits, dim = -1)
            
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id

        logger.log_per_step(input_ids, predictions, labels, loss.item())
        
        pbar.set_postfix(
            loss = round(np.mean(logger.losses), 4),
            **{k: round(v, 4) for k, v in logger.mean_content.items()}
        )
    
logger.end_batch()

for k, v in logger.content.items():
    print(f"{k}: {v}")

joblib.dump(logger.content, folder + "test_metrics.joblib")
joblib.dump(logger.outputs, folder + "test_outputs.joblib")