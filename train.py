from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import torch
from transformers import Trainer, TrainingArguments
from my_tools import *
from my_dataset import *


# get config
config = MyConfig.load_json(sys.argv[1])


# get model
model_name = "Salesforce/blip2-flan-t5-xl"

processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype = torch.float16,
    load_in_8bit = True
)

model.vision_model.requires_grad_(False)
model.qformer.requires_grad_(False)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 64,
    target_modules = ["q", "v", "k", "o"],
    lora_dropout = 0.1,
    bias = "none",
    task_type = TaskType.CAUSAL_LM  # can use SEQ_2_SEQ_LM
)
model.language_model = get_peft_model(model.language_model, lora_config)


# get data
train_ds, val_ds = load_data(
    processor, 
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
    train_ratio = config["dataset"]["train_ratio"], 
    complexities = config["dataset"]["complexities"]
)


# train
training_args = TrainingArguments(
    output_dir = f"save_{config["dir"]}",
    per_device_train_batch_size = config["batch_size"],
    per_device_eval_batch_size = config.get("val_batch_size", config["batch_size"]),
    gradient_accumulation_steps = config["grad_accum"],
    evaluation_strategy = "steps",
    save_strategy = "steps",
    save_steps = 100,
    logging_steps = 10,
    num_train_epochs = config["epochs"],
    learning_rate = config["lr"],
    fp16 = True,
    load_best_model_at_end = True,
    report_to = "none"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    tokenizer = processor.tokenizer
)

trainer.train()