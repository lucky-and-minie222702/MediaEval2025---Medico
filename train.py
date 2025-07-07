from load_dataset import *
from models import *
from torch import optim
from torch.optim import lr_scheduler

epochs = 100
batch_size = 16

train_dl, test_dl, val_dl, tokenizer = load_saved_data(batch_size = batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = tokenizer.all_vocab_size + 1
model = MyGRUModel(vocab_size)

criterion = nn.CrossEntropyLoss(ignore_index = 0)  # ignore padding
optimizer = optim.Adam(model.parameters(), lr = 0.0008)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode = "min",
    factor = 0.1,
    patience = 10,
    min_lr = 1e-5,
)


get_padding_mask = lambda x: torch.clamp(x, max = 1)

def test_before_train():
    with torch.no_grad():
        model.eval()
        for img, ques_ids, ans_ids in train_dl:
            img = img.to(device)
            ques_ids = ques_ids.to(device)
            ans_ids = ans_ids.to(device)
            
            prediction = model(
                image = img,
                questions = ques_ids,
                max_answer_length = 50,
                question_padding_masks = get_padding_mask(ques_ids),
                answers = ans_ids,
                teacher_forcing_ratio = 0.5,
            )  # (B, 50, vocab_size)
            
            prediction = prediction.contiguous().view(-1, vocab_size)  # (B * text_len, vocab_size)
            ans_ids = ans_ids.contiguous().view(-1)  # (B * text_len,)
            
            loss = criterion(prediction, ans_ids)
            break

    torch.cuda.empty_cache()
    print("Test passed!")

# TRAIN

model.to(device)

overall_val_losses = []
overall_train_losses = []

test_before_train()


# DONE

torch.cuda.empty_cache()