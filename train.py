from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import LoraConfig, TaskType
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from my_tools import *
from my_dataset import *
from transformers import logging
torch.set_float32_matmul_precision("high")

os.makedirs("results", exist_ok = True)


# load config
config = MyConfig.load_json(sys.argv[1])

if config.get("debug_mode", False):
    logging.set_verbosity_debug()


# load model
model_name = "Salesforce/instructblip-flan-t5-xxl"
model_path = f"results/{config['dir']}"

processor = InstructBlipProcessor.from_pretrained(model_name)
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
)
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
model.add_adapter(lora_config, "lora")
model.enable_adapters()
model.gradient_checkpointing_disable()
MyUtils.print_trainable_params(model)

# load dataset
train_ds, val_ds = load_data(
    processor, 
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
    train_ratio = config["dataset"]["train_ratio"],
    train_complexities = config["dataset"]["complexities"]
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
    dataloader_pin_memory = False,

    disable_tqdm = not config["tqdm"],
    logging_first_step = True,
)

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