from transformers import Trainer

class SafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs)