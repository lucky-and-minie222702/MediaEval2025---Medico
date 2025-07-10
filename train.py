import joblib
from custom_obj import *
from my_dataset import *
from transformers import BlipForQuestionAnswering, BlipProcessor
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_float32_matmul_precision("high")


batch_size = MyCLI.get_arg("batch_size", 16)
epochs = MyCLI.get_arg("batch_size", 20)
use_tqdm = "tqdm" in sys.argv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
optimizer = Adam(model.parameters(), lr = 1e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.2, patience = 3)
early_stopping_patience = 5

train_dl, val_dl, _ = load_data(processor, batch_size = batch_size)

tqdm_wrapper = lambda dl, name: tqdm(dl, desc = name, ncols = 100, disable = use_tqdm)

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
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            pixel_values = pixel_values,
            labels = labels,
        )

        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        
        pbar.set_postfix(cur_loss = train_losses[-1], avg_loss = np.mean(train_losses))

    # val
    with torch.no_grad():
        model.eval()
        pbar = tqdm_wrapper(val_dl, " Val  ")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                labels = labels,
            )

            loss = outputs.loss

            val_losses.append(loss.item())
            
            pbar.set_postfix(cur_loss = val_losses[-1], avg_loss = np.mean(val_losses))

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    lr_scheduler.step(val_loss)
            
    if not use_tqdm:
        print(f" Train loss : {train_loss}")
        print(f" Val loss   : {val_loss}")
        
    # early stopping:
    if len(overall_val_losses) > early_stopping_patience:
        if min(overall_val_losses[:-early_stopping_patience:]) < min(overall_val_losses[-early_stopping_patience::]):
            print("Early stop triggered!")
            break

torch.save(model, "models/model.torch")
joblib.dump(overall_train_losses, "models/train_loss.joblib")
joblib.dump(overall_val_losses, "models/val_loss.joblib")
