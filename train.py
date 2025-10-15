from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import torch
import pandas as pd
from dataset import CausalDataset
from utils import *
import sys
from itertools import chain
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.optim import AdamW


config = load_json(sys.argv[1])
seed_everything(config["seed"])


pretrained_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_name,
    dtype = torch.bfloat16,
    device_map = "auto",
    trust_remote_code = True
)
processor = Qwen2_5_VLProcessor.from_pretrained(
    pretrained_name, 
    trust_remote_code = True
)
lora_config = LoraConfig(**config["lora_args"])
model = get_peft_model(model, lora_config)


train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/train.csv")
drop_invalid_char_df(train_df)
drop_invalid_char_df(test_df)
df_distributor = DFDistributor(
    train_df = train_df,
    test_df = test_df,
    n_splits = config["n_splits"],
    seed = config["seed"]
)


train_ds = CausalDataset(
    df = df_distributor.fold[config["fold_idx"]]["train"],
    processor = processor,
    mode = "train",
    **config["train_ds_args"],
)
val_ds = CausalDataset(
    df = df_distributor.fold[config["fold_idx"]]["val"],
    processor = processor,
    mode = "both",
    **config.get("val_ds_args", config["train_ds_args"])
)
train_dl = get_dataloader(
    train_ds,
    batch_size = config["train_batch_size"],
    shuffle = True,
)
val_dl = get_dataloader(
    train_ds,
    batch_size = config["val_batch_size"],
    shuffle = False,
)


epochs = config["epochs"]
repeated_train_dl = chain.from_iterable([train_dl] * epochs)


eval_steps = config["eval_steps"]
log_steps = config["log_steps"]
save_limits = config["save_limits"]
accumulation_steps = config["grad_accum"]
tqdm_ncols = 100


train_pbar = tqdm(
    enumerate(repeated_train_dl, 1),
    ncols = tqdm_ncols
)
val_pbar = tqdm(
    enumerate(val_dl),
    ncols = tqdm_ncols
)
dict_to_device = lambda x, d: {k: v.to(d) for k, v in x.items()}


optimizer = AdamW(model.parameters(), lr = config["lr_start"])
lr_scheduler = get_linear_schedule_with_end(
    optimizer, 
    len(repeated_train_dl), 
    config["lr_start"],
    config["lr_end"]
)


logs = []
train_logs = []
train_running_loss = []
train_running_acc = []
for step, batch in train_pbar:
    
    model.train()
    batch = dict_to_device(batch, model.device)
    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / accumulation_steps
    loss.backward()   
    
    train_running_loss.append(loss.item() * accumulation_steps)
    train_running_acc.append(compute_token_accuracy(outputs.logits, batch["labels"]))
    
    if step % accumulation_steps == 0:
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
        
        train_pbar.set_postfix(
            loss = train_running_loss[-1],
            acc = train_running_acc[-1],
        )
        
        if step % log_steps == 0:
            logs = {
                "loss": np.mean(train_running_loss),
                "token_acc": np.mean(train_running_acc),
                "lr": optimizer.param_groups[0]["lr"],
                "step": step
            }

            train_logs.append(logs)
            logs_text = " ".join([f"{k}: {round(v, 4)}" for k, v in logs.items()])

            train_running_loss = []
            train_running_acc = []
        
   
    # save_model_checkpoint(model, config["save_dir"], step, config["save_limit"])

torch.cuda.empty_cache()