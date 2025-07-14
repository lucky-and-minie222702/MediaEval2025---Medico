import joblib
from custom_obj import *
from my_dataset import *
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_float32_matmul_precision("high")

# training config
batch_size = int(MyCLI.get_arg("batch_size", 32))
epochs = int(MyCLI.get_arg("epochs", 20))
use_tqdm = "tqdm" in sys.argv

# models and training strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
optimizer = Adam(model.parameters(), lr = 1e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.25, patience = 3, min_lr = 2e-6)
early_stopping_patience = 5

# data
train_dl, val_dl, _ = load_data(processor, batch_size = batch_size)

# logger
tqdm_wrapper = lambda dl, name: tqdm(dl, desc = name, ncols = 150, disable = not use_tqdm)
val_metric_logger = MyUtils.MetricLogger(processor)
overall_train_losses = []
overall_val_losses = []

# train
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
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        
        pbar.set_postfix(
            loss = round(np.mean(train_losses), 4),
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
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            
            val_losses.append(loss.item())
            val_metric_logger.log_per_step(predictions, labels)
            
            pbar.set_postfix(
                loss = round(np.mean(val_losses), 4),
                **{k: round(v, 4) for k, v in val_metric_logger.mean_content}
            )


    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    lr_scheduler.step(val_loss)
            
    if not use_tqdm:
        print(f" Train loss : {train_loss}")
        print(f" Val loss   : {val_loss}")
    
    overall_train_losses.append(train_loss)
    overall_val_losses.append(val_loss)

    val_metric_logger.end_batch()
        
    # early stopping:
    if len(overall_val_losses) > early_stopping_patience:
        if min(overall_val_losses[:-early_stopping_patience:]) < min(overall_val_losses[-early_stopping_patience::]):
            print("Early stop triggered!")
            break

# save
torch.save(model.state_dict(), "models/model.torch")
joblib.dump(overall_train_losses, "models/train_loss.joblib")
joblib.dump(overall_val_losses, "models/val_loss.joblib")
joblib.dump(val_metric_logger.content, "models/val_metrics.joblib")