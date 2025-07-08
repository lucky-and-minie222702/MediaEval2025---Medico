from load_dataset import *
from custom_obj import *

download_nltk()
# train: 114_868
load_and_save_data(
    question_max_length = 20,  # 114_729 in train
    answer_max_length = 40,  # 113_865 in train
)
