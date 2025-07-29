import joblib
from my_tools import *
from my_dataset import *
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_models import *

torch.set_float32_matmul_precision("high")

# load config
config = MyConfig(sys.argv[1])

# training config
batch_size = config["batch_size"]
epochs = config["epochs"]
use_tqdm = config["use_tqdm"]

# models and training strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train on: {device}")

model, processor = get_models_by_name(config["model"])

if config["use_pretrained"]:
    model.load_state_dict(torch.load(config["pretrained_path"], map_location = device))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr = config["lr"])

lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = config["lr_scheduler"]["factor"], patience = config["lr_scheduler"]["patience"], min_lr = config["lr_scheduler"]["min_lr"])
early_stopping_patience = config["early_stopping"]["patience"]

# data
train_dl, val_dl = load_data(
    processor, 
    batch_size = batch_size, 
    max_question_length = config["dataset"]["mql"], 
    max_answer_length = config["dataset"]["mal"], 
    train_ratio = config["train_ratio"], 
    use_original = config["dataset"]["use_original"], 
    complexities = config["dataset"]["complexities"]
)

# logger
tqdm_wrapper = lambda dl, name, ep: tqdm(dl, 
                                         desc = f" [{ep}] {name}", 
                                         ncols = 150, 
                                         bar_format = "{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                                         disable = not use_tqdm)
val_metric_logger = MyUtils.MetricLogger(processor)
train_metric_logger = MyUtils.MetricLogger(processor)
overall_train_losses = []
overall_val_losses = []

# save path
folder = f"checkpoint_{config['dir_name']}/"
os.makedirs(folder , exist_ok = True)

# train
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    
    train_losses = []
    val_losses = []
    
    # train
    model.train()
    pbar = tqdm_wrapper(train_dl, "Train", e+1)
    for batch in pbar:
        optimizer.zero_grad()
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
        loss.backward()
        optimizer.step()
        
        predictions = model.generate(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            labels = labels,
            max_new_tokens = config["dataset"]["mal"],
        )

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
        for batch in pbar:
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
            val_losses.append(loss.item())
            
            predictions = model.generate(
                input_ids = input_ids,
                pixel_values = pixel_values,
                attention_mask = attention_mask,
                labels = labels,
                max_new_tokens = config["dataset"]["mal"],
            )
            
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
            torch.save(model.state_dict(), folder + f"model_{config["name"]}.torch")
            print("Checkpoint saved!")
            
    print(f"  Train loss : {train_loss}")
    print(f"  Val loss   : {val_loss}")
    print(f"  Lr: {np.mean(lr_scheduler.get_last_lr())}")
    
    overall_train_losses.append(train_loss)
    overall_val_losses.append(val_loss)

    train_metric_logger.end_batch()
    val_metric_logger.end_batch()
        
    # save metrics
    joblib.dump(overall_train_losses, folder + f"train_loss_{config["name"]}.joblib")
    joblib.dump(overall_val_losses, folder + f"val_loss_{config["name"]}.joblib")
    joblib.dump(train_metric_logger.content, folder + f"train_metrics_{config["name"]}.joblib")
    joblib.dump(val_metric_logger.content, folder + f"val_metrics_{config["name"]}.joblib")
        
    # early stopping
    if len(overall_val_losses) > early_stopping_patience:
        if min(overall_val_losses[:-early_stopping_patience:]) < min(overall_val_losses[-early_stopping_patience::]):
            print("Early stop triggered!")
            break
        
if epochs == 1: 
    torch.save(model.state_dict(), folder + f"model_{config["name"]}.torch")