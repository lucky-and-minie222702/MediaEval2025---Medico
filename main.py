from config import *
import sys

conf = load_json(sys.argv[1])

env, model_interface, class_conf = get_env(conf)
env.train(
    fold_idx = conf["fold_idx"],
    do_test = conf["do_test"],
    format_data_fn = class_conf["format_data_fn"],
    
    train_ds_args = conf["train_ds_args"],
    val_ds_args = conf.get("val_ds_args"),
    test_ds_args = conf.get("test_ds_args"),
    
    test_batch_size = conf["test_batch_size"],
    
    generation_conf = conf.get("generation_conf"),
)

