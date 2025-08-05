from transformers import Blip2Processor, Blip2ForConditionalGeneration
from my_tools import *
from my_dataset import *
from torch.utils.data import DataLoader
import torch


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
model_path = f"results/{config['dir']}"
model = Blip2ForConditionalGeneration.from_pretrained(model_path)
processor = Blip2Processor.from_pretrained(model_path)


# load dataset
test_ds = load_data(
    processor, 
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
    fold = config["dataset"]["fold"], 
    test_complexities = config["dataset"]["complexities"],
    test_only = True
)
test_dl = DataLoader(
    test_ds, 
    batch_size = config["batch_size"],
    num_workers = 4,
)

# test
with torch.no_grad():
    