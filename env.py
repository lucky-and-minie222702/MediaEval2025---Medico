from base import *
from transformers import Trainer, TrainingArguments

class TrainingEnvironment:
    def __init__(
        self, 
        train_df, 
        test_df, 
        n_splits, 
        model_interface: ModelInterface,
        dataset_class = None,
        training_args = {},
        seed = 27022009,
    ):
        seed_everything(seed)

        self.distributor = DFDistributor(train_df, test_df, n_splits, seed)
        self.dataset_class = dataset_class
        if self.dataset_class is None:
            self.dataset_class = BaseDataset
        self.model_interface = model_interface
        
        self.training_arguments = TrainingArguments(**training_args)
        self.trainer = None
        
        self.model_interface.model.config.use_cache = False
        self.trainer.model_accepts_loss_kwargs = False
    
    def get_train(self, fold_idx, **kwargs):
        return self.dataset_class(
            df = self.distributor.fold[fold_idx][0],
            processor = self.model_interface.processor,
            **kwargs
        )
        
    def get_val(self, fold_idx, **kwargs):
        return self.dataset_class(
            df = self.distributor.fold[fold_idx][1],
            processor = self.model_interface.processor,
            **kwargs
        )
        
    def get_test(self, **kwargs):
        return self.dataset_class(
            df = self.distributor.test_df,
            processor = self.model_interface.processor,
            mode = "infer",
            **kwargs
        )
        
    def train(
        self, 
        fold_idx, 
        
        do_test = True,
        
        train_ds_args = {},
        val_ds_args = None,
        test_ds_args = None,
        
        lora_args = {},
        
        test_batch_size = 16,
        format_data_fn = None,
        generation_conf = None,
    ):
        if val_ds_args is None:
            val_ds_args = train_ds_args
        if test_ds_args is None:
            test_ds_args = train_ds_args

        train_ds = self.get_train(fold_idx, **train_ds_args)
        val_ds = self.get_val(fold_idx, **val_ds_args)
        
        self.model_interface.to_lora(**lora_args)
        
        self.trainer = Trainer(
            model = self.model_interface.model,
            processing_class = self.model_interface.processor,
            
            train_dataset = train_ds,
            eval_dataset = val_ds,
            
            callbacks = [ModelUtils.TrainerSaveLossCallback(self.training_arguments.output_dir)]
        )
        
        self.trainer.model_accepts_loss_kwargs = False
        self.trainer.train()
        
        if do_test:
            test_ds = self.get_test(**test_ds_args)
            test_dl = get_dataloader(
                test_ds,
                shuffle = False,
                batch_size = test_batch_size,
            )
            self.model_interface.test(
                dl = test_dl,
                output_dir = self.training_arguments.output_dir,
                generation_config = generation_conf,
                format_data_fn = format_data_fn,
            )