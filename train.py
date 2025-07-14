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
lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 2)
early_stopping_patience = 3

train_dl, val_dl, _ = load_data(processor, batch_size = batch_size)

tqdm_wrapper = lambda dl, name: tqdm(dl, desc = name, ncols = 150, disable = not use_tqdm)

def get_bleu_score(label, pred):
    label = label.detach().cpu().numpy().tolist()
    pred = pred.detach().cpu().numpy().tolist()
    
    label = processor.tokenizer.batch_decode(pred, skip_special_tokens = True)    
    pred = processor.tokenizer.batch_decode(pred, skip_special_tokens = True)
    
    label = list(map(lambda s: s.split(), label))
    pred = list(map(lambda s: s.split(), pred))
    
    return MyText.bleu_score_batch(label, pred)


overall_train_losses = []
overall_val_losses = []

overall_train_bleus = []
overall_val_bleus = []

for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    
    train_losses = []
    val_losses = []
    
    train_bleus = []
    val_bleus = []
    
    # train
    model.train()
    pbar = tqdm_wrapper(train_dl, " Train")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        prediction = torch.argmax(outputs.logits, dim = -1)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        train_bleus.append(get_bleu_score(batch["labels"], prediction))
        
        pbar.set_postfix(
            avg_loss = np.mean(train_losses),
            avg_bleu = np.mean(train_bleus)
        )

    # val
    with torch.no_grad():
        model.eval()
        pbar = tqdm_wrapper(val_dl, " Val  ")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            prediction = torch.argmax(outputs.logits, dim = -1)
            
            val_losses.append(loss.item())
            val_bleus.append(get_bleu_score(batch["labels"], prediction))
            
            pbar.set_postfix(
                avg_loss = np.mean(val_losses),
                avg_bleu = np.mean(val_bleus)
            )


    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    
    train_bleu = np.mean(train_bleus)
    val_bleu = np.mean(val_bleus)

    lr_scheduler.step(val_loss)
            
    if not use_tqdm:
        print(f" Train loss : {train_loss}")
        print(f" Val loss   : {val_loss}")
        print(f" Train bleu : {train_bleu}")
        print(f" Val bleu   : {val_bleu}")
        
    
    overall_train_losses.append(train_loss)
    overall_val_losses.append(val_loss)
    
    overall_train_bleus.append(train_bleu)
    overall_val_bleus.append(val_bleu)
        
        
    # early stopping:
    if len(overall_val_losses) > early_stopping_patience:
        if min(overall_val_losses[:-early_stopping_patience:]) < min(overall_val_losses[-early_stopping_patience::]):
            print("Early stop triggered!")
            break

torch.save(model, "models/model.torch")
joblib.dump(overall_train_losses, "models/train_loss.joblib")
joblib.dump(overall_val_losses, "models/val_loss.joblib")
