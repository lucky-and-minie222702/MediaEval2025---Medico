from huggingface_hub import HfApi
from my_tools import *
from my_dataset import *
from my_tools import *
from my_dataset import *


# load config
config = MyConfig.load_json(sys.argv[1])

checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))
model_path = f"results/{config['dir']}/checkpoint-{checkpoint}"

repo_id = "trietbui/instructblip-flan-t5-xxl-kvasir-vqa-x1"

api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)

api.upload_folder(
    repo_id = repo_id,
    folder_path = model_path,
    repo_type = "model",
)
