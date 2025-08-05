from transformers import Blip2Processor, Blip2ForConditionalGeneration
from my_tools import *


# load config
config = MyConfig.load_json(sys.argv[1])


# load model
model_path = f"results/{config['dir']}"

model = Blip2ForConditionalGeneration.from_pretrained(model_path)
processor = Blip2Processor.from_pretrained(model_path)

