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
config = MyConfig("train_config.json")

# training config
batch_size = config["batch_size"]
epochs = config["epochs"]
use_tqdm = config["use_tqdm"]

# models and training strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")
model, processor = get_models_by_name(config["model_name"])
model = model.to(device)
optimizer = Adam(model.parameters(), lr = config["lr"])
lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = config["lr_scheduler"]["factor"], patience = config["lr_scheduler"]["patience"], min_lr = config["lr_scheduler"]["min_lr"])
early_stopping_patience = config["early_stopping"]["patience"]

# data
train_dl, val_dl, _ = load_data(processor, max_question_length = config["dataset"]["mql"], max_answer_length = config["dataset"]["mal"], train_ratio = config["train_ratio"], batch_size = batch_size, use_original = config["dataset"]["use_original"])

# logger
tqdm_wrapper = lambda dl, name, ep: tqdm(dl, desc = f" [{ep}] {name}", ncols = 175, disable = not use_tqdm)
val_metric_logger = MyUtils.MetricLogger(processor)
train_metric_logger = MyUtils.MetricLogger(processor)
overall_train_losses = []
overall_val_losses = []

# save path
folder = f"models_checkpoint_{config["name"]}/"
os.makedirs(folder , exist_ok = True)

# train
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    
    train_losses = []
    val_losses = []
    
    # train
    model.train()
    pbar = tqdm_wrapper(train_dl, "Train", e+1)
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        optimizer.zero_grad()

        loss = outputs.loss
        predictions = torch.argmax(outputs.logits, dim = -1)
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_metric_logger.log_per_step(predictions, labels)
        
        pbar.set_postfix(
            loss = round(np.mean(train_losses), 4),
            **{k: round(v, 4) for k, v in train_metric_logger.mean_content.items()}
        )

    # val
    with torch.no_grad():
        model.eval()
        pbar = tqdm_wrapper(val_dl, "Val  ", e+1)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim = -1)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            
            val_losses.append(loss.item())
            val_metric_logger.log_per_step(predictions, labels)
            
            pbar.set_postfix(
                loss = round(np.mean(val_losses), 4),
                **{k: round(v, 4) for k, v in val_metric_logger.mean_content.items()}
            )


    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    lr_scheduler.step(val_loss)
    
    if len(overall_val_losses) > 0:
        if val_loss < min(overall_val_losses):
            torch.save(model.state_dict(), folder + "model.torch")
            print("Checkpoint saved!")
            
    print(f"  Train loss : {train_loss}")
    print(f"  Val loss   : {val_loss}")
    
    overall_train_losses.append(train_loss)
    overall_val_losses.append(val_loss)

    train_metric_logger.end_batch()
    val_metric_logger.end_batch()
        
    # save metrics
    joblib.dump(overall_train_losses, folder + "train_loss.joblib")
    joblib.dump(overall_val_losses, folder + "val_loss.joblib")
    joblib.dump(train_metric_logger.content, folder + "train_metrics.joblib")
    joblib.dump(val_metric_logger.content, folder + "val_metrics.joblib")
        
    # early stopping:
    if len(overall_val_losses) > early_stopping_patience:
        if min(overall_val_losses[:-early_stopping_patience:]) < min(overall_val_losses[-early_stopping_patience::]):
            print("Early stop triggered!")
            break
        
if epochs == 1:
    torch.save(model.state_dict(), folder + "model.torch")