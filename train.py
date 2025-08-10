from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, TaskType
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from my_tools import *
from my_dataset import *
from transformers import logging
logging.set_verbosity_debug()

os.makedirs("results", exist_ok = True)


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
model_name = config["model_name"]
model_path = f"results/{config['dir']}"

processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype = torch.bfloat16,
)

lora_config = LoraConfig(
    r = config["lora"]["r"],
    lora_alpha = config["lora"]["alpha"],
    target_modules = [
        # q-former
        "q_former.encoder.layer.*.attention.self.query",
        "q_former.encoder.layer.*.attention.self.key",
        "q_former.encoder.layer.*.attention.self.value",
        "q_former.encoder.layer.*.attention.output.dense",
        "q_former.encoder.layer.*.intermediate.dense",
        "q_former.encoder.layer.*.output.dense",
        # language model
        "language_model.model.layers.*.self_attn.q_proj",
        "language_model.model.layers.*.self_attn.k_proj",
        "language_model.model.layers.*.self_attn.v_proj",
        "language_model.model.layers.*.self_attn.o_proj",
        # mlp:
        "language_model.model.layers.*.mlp.gate_proj",
        "language_model.model.layers.*.mlp.up_proj",
        "language_model.model.layers.*.mlp.down_proj",
    ],
    modules_to_save = [
        "language_projection",
        "language_model.lm_head",
    ],
    lora_dropout = config["lora"].get("dropout", 0.0),
    inference_mode = False,
    bias = "none",
    task_type = TaskType.QUESTION_ANS
)
model.add_adapter(lora_config, adapter_name = "lora_1")
model.enable_adapters()


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
    eval_accumulation_steps = config.get("val_grad_accum", config["grad_accum"]),
    
    eval_strategy = "steps",
    eval_steps = config["n_steps"],
    
    save_strategy = "steps",
    save_steps = config["n_steps"],
    
    metric_for_best_model = "eval_loss",
    save_total_limit = 1,
    
    logging_strategy = "steps",
    logging_steps = config["log_steps"],
    
    fp16 = False,
    bf16 = True,
    
    report_to = "none",
    
    dataloader_num_workers = 4,
    dataloader_persistent_workers = True,
    dataloader_pin_memory = False,
    
    remove_unused_columns = False,

    disable_tqdm = not config["tqdm"],
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