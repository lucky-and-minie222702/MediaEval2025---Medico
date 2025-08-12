import joblib
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from my_tools import *
from my_dataset import *
import torch
from tqdm import tqdm
import pandas as pd
from my_tools import *
from my_dataset import *


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
model_path = f"results/{config['dir']}/checkpoint-{MyUtils.get_latest_checkpoint(config['dir']) if config['checkpoint'] == 'latest' else config['checkpoint']}"
file_path = f"{model_path}-test.results"  # for save test results files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Blip2ForConditionalGeneration.from_pretrained(
    model_path,
    device_map = "auto",
    torch_dtype = torch.bfloat16,
).to(device)
processor = Blip2Processor.from_pretrained(model_path)


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
            
            do_sample = config["gen"].get("do_sample", True),
            max_new_tokens = config["dataset"]["max_answer_length"],
            num_beams = config["gen"]["n_beams"],
            early_stopping = config["gen"].get("early_stopping", False),
            num_return_sequences = config["gen"]["n_returns"],
            temperature = config["gen"].get("temperature", 1.0),
        )
        
        logger.log_per_step(
            quest = batch["input_ids"],
            pred = predictions,
            label = labels,
            n_returns = config["gen"]["n_returns"],
        )
        
        pbar.set_postfix(**logger.cur_scores)
logger.end()
joblib.dump(logger.results, file_path)
        