import joblib
from my_tools import *
from my_dataset import *
import torch
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
criterion = get_loss_by_name(config["loss"], processor)

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
        
        predictions = torch.argmax(outputs.logits, dim = -1)

        logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)
        loss = loss.view(config["batch_size"], config["dataset"]["mal"])
        
        sample_w = batch["weights"].unsqueeze(-1)
        loss = (loss * sample_w).mean()

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