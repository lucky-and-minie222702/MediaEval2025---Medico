import joblib
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from my_tools import *
from my_dataset import *
import torch
from tqdm import tqdm
import pandas as pd
from my_tools import *
from my_dataset import *
torch.set_float32_matmul_precision("high")


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))
model_path = f"results/{config['dir']}/checkpoint-{checkpoint}"
file_path = f"{model_path}-test.results"  # for save test results files

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config = bnb_4bit,
).to(device)
processor = InstructBlipProcessor.from_pretrained(model_path)


# load dataset
test_ds = load_data(
    processor, 
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
    test_complexities = config["dataset"]["complexities"],
    test_only = True
)
test_dl = MyUtils.get_dataloader(test_ds, batch_size = config["batch_size"], shuffle = False)


# test
test_df = pd.read_csv("data/test.csv")
logger = MyUtils.TestLogger(processor)
with torch.no_grad():
    pbar = tqdm(test_dl)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        labels = batch.pop("labels", None)
        
        predictions = model.generate(
            **batch,
            
            do_sample = config["gen"].get("do_sample", False),
            max_new_tokens = config["dataset"]["max_answer_length"],
            num_beams = config["gen"].get("n_beams", 1),
            early_stopping = config["gen"].get("early_stopping", False),
            num_return_sequences = config["gen"].get("n_returns", 1),
            
            **config["gen"].get("others", {})
        )
        
        logger.log_per_step(
            quest = batch["input_ids"],
            pred = predictions,
            label = labels,
            n_returns = config["gen"].get("n_returns", 1),
        )
        
        pbar.set_postfix(**logger.cur_scores)
logger.end()
joblib.dump(logger.results, file_path)
        