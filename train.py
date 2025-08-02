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

if config["use_prefinetuned"]:
    model.load_state_dict(torch.load(config["prefinetuned_path"], map_location = device))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr = config["lr"])

lr_scheduler = ReduceLROnPlateau(optimizer, 
                                 mode = "min", 
                                 factor = config["lr_scheduler"]["factor"], 
                                 patience = config["lr_scheduler"]["patience"], 
                                 min_lr = config["lr_scheduler"]["min_lr"])

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
                                         bar_format = "{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]",
                                         disable = not use_tqdm)
val_metric_logger = MyUtils.MetricLogger(processor, early_stopping_patience = config["early_stopping"]["patience"])
train_metric_logger = MyUtils.MetricLogger(processor)

# save path
folder = f"checkpoint_{config['dir_name']}/"
os.makedirs(folder , exist_ok = True)

# train
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    
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

        train_metric_logger.log_per_step(predictions, labels, loss.item())
        
        pbar.set_postfix(
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
            
            predictions = model.generate(
                input_ids = input_ids,
                pixel_values = pixel_values,
                attention_mask = attention_mask,
                labels = labels,
                max_new_tokens = config["dataset"]["mal"],
            )
            
            val_metric_logger.log_per_step(predictions, labels, loss.item())
            
            pbar.set_postfix(
                **{k: round(v, 4) for k, v in val_metric_logger.mean_content.items()}
            )


    train_loss = train_metric_logger.mean_content["loss"]
    val_loss = val_metric_logger.mean_content["loss"]
    
    lr_scheduler.step(val_loss)

    if e > 0:
        if val_loss < min(val_metric_logger.batch_content["loss"]):
            torch.save(model.state_dict(), folder + f"model_{config['name']}.torch")
            print("Checkpoint saved!")
            
    print(f"  Train loss : {train_loss}")
    print(f"  Val loss   : {val_loss}")
    print(f"  Lr         : {optimizer.param_groups[0]['lr']}")

    train_metric_logger.end_batch()
    val_metric_logger.end_batch()
        
    # save metrics
    joblib.dump(train_metric_logger.content, folder + f"train_metrics_{config['name']}.joblib")
    joblib.dump(val_metric_logger.content, folder + f"val_metrics_{config['name']}.joblib")
        
    # early stopping
    if val_metric_logger.is_early_stop():
        print("Early stop triggered!")
        break
        
if epochs == 1: 
    torch.save(model.state_dict(), folder + f"model_{config['name']}.torch")