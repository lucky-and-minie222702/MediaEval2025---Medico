from transformers import Blip2Processor, Blip2ForConditionalGeneration, GenerationConfig
from my_tools import *
from my_dataset import *
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
model_path = f"results/{config['dir']}/checkpoint-{config['checkpoint']}"
file_path = f"{model_path}-test.results"  # for save test results files
model = Blip2ForConditionalGeneration.from_pretrained(model_path)
processor = Blip2Processor.from_pretrained(model_path)


# load dataset
test_ds = load_data(
    processor, 
    max_question_length = config["dataset"]["max_question_length"], 
    max_answer_length = config["dataset"]["max_answer_length"], 
    test_complexities = config["dataset"]["complexities"],
    test_only = True
)
test_dl = MyUtils.get_dataloader(test_ds, batch_size = config["batch_size"])

# test
with torch.no_grad():
    for batch in tqdm(test_dl):
        labels = batch.pop("labels", None)
        
        predictions = model.generate(
            **batch,
            
            do_sample = True,
            max_new_tokens = config["dataset"]["max_answer_length"],
            num_beams = config["gen"]["n_beams"],
            early_stopping = True,
            num_return_sequences = config["gen"]["n_returns"],
        )
        predictions = MyUtils.torch_to_list(predictions)
        predictions = processor.tokenizer.batch_decode(predictions, skip_special_tokens = True)
        
        labels = MyUtils.torch_to_list(labels)
        labels = processor.tokenizer.batch_decode(labels, skip_special_tokens = True)
        
        print(labels)
        print(predictions)
        break
        