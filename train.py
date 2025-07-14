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
epochs = int(MyCLI.get_arg("epochs", 20))
use_tqdm = "tqdm" in sys.argv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
optimizer = Adam(model.parameters(), lr = 1e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.25, patience = 3, min_lr = 2e-6)
early_stopping_patience = 5

train_dl, val_dl, _ = load_data(processor, batch_size = batch_size)

tqdm_wrapper = lambda dl, name: tqdm(dl, desc = name, ncols = 150, disable = not use_tqdm)


train_metric_logger = MyUtils.MetricLogger(processor)
val_metric_logger = MyUtils.MetricLogger(processor)


overall_train_losses = []
overall_val_losses = []

for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    
    train_losses = []
    val_losses = []
    
    # train
    model.train()
    pbar = tqdm_wrapper(train_dl, " Train")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        optimizer.zero_grad()

        loss = outputs.loss
        predictions = torch.argmax(outputs.logits, dim = -1)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        train_metric_logger.log_per_step(predictions, batch["labels"])
        
        pbar.set_postfix(
            loss = round(np.mean(train_losses), 4),
            meteor = round(train_metric_logger.mean_content["meteor"], 4),
            bleu = round(train_metric_logger.mean_content["bleu"], 4),
        )
        break

    # val
    with torch.no_grad():
        model.eval()
        pbar = tqdm_wrapper(val_dl, " Val  ")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim = -1)
            
            val_losses.append(loss.item())
            val_metric_logger.log_per_step(predictions, batch["labels"])
            
            pbar.set_postfix(
                loss = round(np.mean(val_losses), 4),
                meteor = round(val_metric_logger.mean_content["meteor"], 4),
                bleu = round(val_metric_logger.mean_content["bleu"], 4),
            )
            break


    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    lr_scheduler.step(val_loss)
            
    if not use_tqdm:
        print(f" Train loss : {train_loss}")
        print(f" Val loss   : {val_loss}")
    
    overall_train_losses.append(train_loss)
    overall_val_losses.append(val_loss)
        
    train_metric_logger.end_batch()
    val_metric_logger.end_batch()
        
    # early stopping:
    if len(overall_val_losses) > early_stopping_patience:
        if min(overall_val_losses[:-early_stopping_patience:]) < min(overall_val_losses[-early_stopping_patience::]):
            print("Early stop triggered!")
            break

torch.save(model.state_dict(), "models/model.torch")

joblib.dump(overall_train_losses, "models/train_loss.joblib")
joblib.dump(overall_val_losses, "models/val_loss.joblib")

joblib.dump(train_metric_logger.content, "models/train_metrics.joblib")
joblib.dump(val_metric_logger.content, "models/val_metrics.joblib")