from load_dataset import *
from models import *
from torch import optim
from torch.optim import lr_scheduler
import sys

os.makedirs("models", exist_ok = True)

def get_arg(name, default = None):
    if name in sys.argv:
        i = sys.argv.index(name)
        try:
            return sys.argv[i+1]
        except:
            return default
    return default

epochs = int(get_arg("epochs", 100))
batch_size = int(get_arg("batch_size", 15))

train_dl, test_dl, val_dl, tokenizer = load_saved_data(batch_size = batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = tokenizer.all_vocab_size + 1
model = MyGRUModel(vocab_size)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index = 0)  # ignore padding
optimizer = optim.Adam(model.parameters(), lr = 0.0008)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode = "min",
    factor = 0.2,
    patience = 5,
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
                max_answer_length = answer_max_length,
                question_padding_masks = get_padding_mask(ques_ids),
                answers = ans_ids,
                teacher_forcing_ratio = 0.5,
            )  # (B, answer_max_length, vocab_size)
            
            prediction = model(
                image = img,
                questions = ques_ids,
                max_answer_length = answer_max_length,
                question_padding_masks = get_padding_mask(ques_ids),
                # answers = ans_ids,
                # teacher_forcing_ratio = 0.5,
            )  # (B, answer_max_length, vocab_size)
            
            prediction = prediction.contiguous().view(-1, vocab_size)  # (B * text_len, vocab_size)
            ans_ids = ans_ids.contiguous().view(-1)  # (B * text_len,)
            
            loss = criterion(prediction, ans_ids)
            break

    torch.cuda.empty_cache()
    print("Test passed!")

# TRAIN

test_before_train()

answer_max_length = 50

# epoch level metrics
overall_val_losses = []
overall_train_losses = []

overall_train_bleu_scores = []
overall_val_bleu_scores = []

for e in range(epochs):
    print(f"Epoch {e+1} / {epochs}")
    
    train_losses = []
    val_losses = []
    
    train_bleu_scores = []
    val_bleu_scores = []
    
    # train
    model.train()
    pbar = tqdm(train_dl, desc = "Train")
    for img, ques_ids, ans_ids in pbar:
        img = img.to(device)
        ques_ids = ques_ids.to(device)
        ans_ids = ans_ids.to(device)
        
        prediction = model(
            image = img,
            questions = ques_ids,
            max_answer_length = answer_max_length,
            question_padding_masks = get_padding_mask(ques_ids),
            answers = ans_ids,
            teacher_forcing_ratio = 0.5,
        )  # (B, answer_max_length, vocab_size)
        
        prediction = prediction.contiguous().view(-1, vocab_size)  # (B * text_len, vocab_size)
        ans_ids = ans_ids.contiguous().view(-1)  # (B * text_len,)
        
        loss = criterion(prediction, ans_ids)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_bleu_scores.append(MyText.bleu_score_batch(ans_ids.cpu().numpy(), prediction.cpu().numpy()))
        
        pbar.set_postfix(loss = np.mean(train_losses), bleu = np.mean(train_bleu_scores))
        
    overall_train_losses.append(np.mean(train_losses))
    overall_train_bleu_scores.append(np.mean(train_bleu_scores))

	
    # val
    with torch.no_grad():
        pbar = tqdm(val_dl, desc = "Val")
        
        model.eval()
        for img, ques_ids, ans_ids in pbar:
            img = img.to(device)
            ques_ids = ques_ids.to(device)
            ans_ids = ans_ids.to(device)
            
            prediction = model(
                image = img,
                questions = ques_ids,
                max_answer_length = answer_max_length,
                question_padding_masks = get_padding_mask(ques_ids),
                # answers = ans_ids,
                # teacher_forcing_ratio = 0.5,
            )  # (B, answer_max_length, vocab_size)
            
            prediction = prediction.contiguous().view(-1, vocab_size)  # (B * text_len, vocab_size)
            ans_ids = ans_ids.contiguous().view(-1)  # (B * text_len,)
            
            loss = criterion(prediction, ans_ids)
            
            val_losses.append(loss.item())
            val_bleu_scores.append(MyText.bleu_score_batch(ans_ids.cpu().numpy(), prediction.cpu().numpy()))
            
            pbar.set_postfix(loss = np.mean(val_losses), bleu = np.mean(val_bleu_scores))
            
    overall_train_losses.append(np.mean(val_losses))
    overall_train_bleu_scores.append(np.mean(val_bleu_scores))
    
    lr_scheduler.step(overall_val_losses)
    
    # early stopping
    patience = 12
    if len(overall_val_losses) > 12:
        if min(overall_val_losses[:-patience:]) < min(overall_val_losses[-patience::]):
            print("Early stopping triggered!")
            break


torch.save(model.state_dict(), "models/baseline_weights.pth")

joblib.dump(overall_train_losses, "models/train_loss.joblib")
joblib.dump(overall_train_bleu_scores, "models/train_bleu.joblib")

joblib.dump(overall_val_losses, "models/val_loss.joblib")
joblib.dump(overall_val_bleu_scores, "models/val_bleu.joblib")

print("Done")

# DONE

torch.cuda.empty_cache()