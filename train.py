from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType
import torch
from transformers import Trainer, Seq2SeqTrainingArguments
from my_tools import *
from my_dataset import *

os.makedirs("results", exist_ok = True)


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
model_name = "Salesforce/blip2-flan-t5-xl"
model_path = f"results/{config['dir']}"

processor = Blip2Processor.from_pretrained(model_name)
quant_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
    llm_int8_skip_modules = None,
    llm_int8_enable_fp32_cpu_offload = True,
)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype = torch.float16,
    quantization_config = quant_config,
)

model.vision_model.requires_grad_(False)
model.qformer.requires_grad_(False)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 64,
    target_modules = ["q", "v", "k", "o"],
    lora_dropout = 0.1,
    inference_mode = False,
    bias = "none",
    task_type = TaskType.CAUSAL_LM  # can use SEQ_2_SEQ_LM
)
model.add_adapter(lora_config, adapter_name="lora_1")
model.enable_adapters()


# load dataset
train_ds, val_ds = load_data(
    processor, 
    train_ratio = config["dataset"]["train_ratio"],
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
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
    
    eval_strategy = "steps",
    eval_steps = config["n_steps"],
    
    save_strategy = "steps",
    save_steps = config["n_steps"],
    
    metric_for_best_model = "eval_loss",
    load_best_model_at_end = True,
    save_total_limit = 1,
    
    logging_strategy = "steps",
    logging_steps = config["log_steps"],
    
    predict_with_generate = True,
    
    fp16 = False,
    bf16 = True,
    report_to = "none",
    
    dataloader_num_workers = 4,
    dataloader_pin_memory = True,
    dataloader_persistent_workers = True,

    disable_tqdm = not config["tqdm"],
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    processing_class = processor,
)
trainer.model_accepts_loss_kwargs = False
trainer.train()