import joblib
from my_tools import *
from my_dataset import *
import torch
from my_models import *

torch.set_float32_matmul_precision("high")

# load config
config = MyConfig(sys.argv[1])

# testing config
batch_size = config["batch_size"]
use_tqdm = config["use_tqdm"]

# models and testing strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Test on: {device}")
model, processor = get_models_by_name(config["model"])

# data
test_dl = load_data(
    processor, 
    batch_size = batch_size, 
    max_question_length = config["dataset"]["mql"], 
    max_answer_length = config["dataset"]["mal"], 
    use_original = config["dataset"]["use_original"], 
    complexities = config["dataset"]["complexities"],
    test_only = True,
)

# logger
logger = MyUtils.TestLogger(processor)

# save path
folder = f"checkpoint_{config['dir_name']}/"
model.load_state_dict(torch.load(folder + f"model_{config['name']}.torch", map_location = device))
model = model.to(device)

tqdm_wrapper = lambda dl, name: tqdm(dl, 
                                     desc = f"{name}", 
                                     ncols = 150, 
                                     bar_format = "{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]",
                                     disable = not use_tqdm)

with torch.no_grad():
    model.eval()
    pbar = tqdm_wrapper(test_dl, "Test")
    for batch, _ in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        outputs = model(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            labels = labels,
        )
        
        loss = outputs.loss
        
        predictions = model.generate(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            labels = labels,
            max_new_tokens = config["dataset"]["mal"],
        )

        logger.log_per_step(input_ids, predictions, labels, loss.item())
        
        pbar.set_postfix(
            loss = round(np.mean(logger.losses), 4),
            **{k: round(v, 4) for k, v in logger.mean_content.items()}
        )
    
logger.end_batch()

for k, v in logger.content.items():
    print(f"{k}: {v}")

joblib.dump(logger.content, folder + f"test_metrics_{config['name']}.joblib")