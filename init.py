import os
from custom_obj import *

os.makedirs("data/save", exist_ok = True)
os.makedirs("data/models", exist_ok = True)

download_nltk()