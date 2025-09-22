from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, TaskType, LoraModel, get_peft_model
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from my_tools import *
from my_dataset import *
from torch import nn
from transformers import logging
import torch.nn.functional as F

os.makedirs("results", exist_ok = True)


# load config
config = MyConfig.load_json(sys.argv[1])

if config.get("debug_mode", False):
    logging.set_verbosity_debug()


# load model
model_name = config["model"]
model_path = f"results/{config['dir']}"

processor = InstructBlipProcessor.from_pretrained(model_name)
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code = True,
    dtype = torch.bfloat16,
)


# mod vision encoder
# assert IMG_SIZE % 16 == 0
# NEW_PATCH = IMG_SIZE // 16

# old_conv = model.vision_model.embeddings.patch_embedding
# W_old = old_conv.weight.data
# b_old = old_conv.bias.data if old_conv.bias is not None else None

# new_conv = nn.Conv2d(
#     3, 
#     W_old.shape[0], 
#     kernel_size = NEW_PATCH, 
#     stride = NEW_PATCH, 
#     bias = b_old is not None,
# )

# W_resized = F.interpolate(
#     W_old, size = (NEW_PATCH, NEW_PATCH), mode = "bicubic", align_corners = False
# )
# scale = (14.0 * 14.0 / (NEW_PATCH * NEW_PATCH)) ** 0.5
# new_conv.weight.data.copy_(W_resized * scale)
# if b_old is not None:
#     new_conv.bias.data.copy_(b_old) 

# model.vision_model.embeddings.patch_embedding = new_conv


lora_config = LoraConfig(
    r = config["lora"]["r"],
    lora_alpha = config["lora"]["alpha"],
    target_modules = [
        "q",
        "k",
        "v",
        "o",
    ],
    lora_dropout = config["lora"].get("dropout", 0.0),
    bias = "none",
    task_type = TaskType.SEQ_2_SEQ_LM
)
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
MyUtils.print_trainable_params(model)


# load dataset
train_ds, val_ds = load_data(
    processor, 
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
    train_ratio = config["dataset"]["train_ratio"],
    train_complexities = config["dataset"]["complexities"],
    train_augment = config["dataset"]["augment"],
    caption_prompt = config["dataset"]["caption_prompt"],
)


# train
training_args = Seq2SeqTrainingArguments(
    output_dir = model_path,
    
    num_train_epochs = config["epochs"],
    learning_rate = config["lr"],
    
    per_device_train_batch_size = config["batch_size"],
    per_device_eval_batch_size = config.get("val_batch_size", config["batch_size"]),

    gradient_accumulation_steps = config["grad_accum"],
    eval_accumulation_steps = config.get("val_accum", None),
    
    eval_strategy = "steps",
    eval_steps = config["val_steps"],
    
    save_strategy = "best",
    metric_for_best_model = "eval_loss",

    save_total_limit = 1,
    
    logging_strategy = "steps",
    logging_steps = config["log_steps"],
    
    lr_scheduler_type = config["lr_scheduler"],
    warmup_steps = config["warmup_steps"],  
    
    bf16 = True,
    
    report_to = "none",
    
    dataloader_num_workers = 4,
    dataloader_persistent_workers = True,
    dataloader_pin_memory = True,

    disable_tqdm = not config["tqdm"],
    logging_first_step = True,
    
    prediction_loss_only = True,
)
model.config.use_cache = False
trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    processing_class = processor,
    callbacks = [TrainerSaveLossCallback(model_path)]
)

trainer.model_accepts_loss_kwargs = False
trainer.train()