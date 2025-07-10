from load_dataset import *
from CustomModules.models import *
from torch import optim
from torch.optim import lr_scheduler
import sys

torch.set_float32_matmul_precision("high")

os.makedirs("models", exist_ok = True)

def get_arg(name, default = None):
    if name in sys.argv:
        i = sys.argv.index(name)
        try:
            return sys.argv[i+1]
        except:
            return default
    return default

answer_max_length = 40

epochs = int(get_arg("epochs", 100))
batch_size = int(get_arg("batch_size", 64))

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
torch_to_list = lambda t: t.cpu().detach().numpy().tolist()

def process(prediction, ans_ids):
    prediction = prediction.contiguous().view(-1, vocab_size)  # (B * text_len, vocab_size)
    ans_ids = ans_ids.contiguous().view(-1)  # (B * text_len) 
    
    loss = criterion(prediction, ans_ids)
    
    prediction = prediction.contiguous().view(-1, answer_max_length, vocab_size)  # (B, text_len, vocab_size)
    prediction = prediction.contiguous().argmax(dim = -1)  # (B, text_len)
    ans_ids = ans_ids.contiguous().view(-1, answer_max_length)  # (B, text_len) 
    
    prediction = torch_to_list(prediction)
    ans_ids = torch_to_list(ans_ids)
    
    prediction = tokenizer.decode_sentence(prediction)
    ans_ids = tokenizer.decode_sentence(ans_ids)
    
    bleu_score = MyText.bleu_score_batch(ans_ids, prediction)
    
    return loss, bleu_score

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
            
            print("Example prediction shape:", prediction.shape)
            print("Example answer shape:", ans_ids.shape)

            assert len(prediction.shape) == 3
            assert len(ans_ids.shape) == 2
            
            loss, bleu_score = process(prediction, ans_ids)

            break

    torch.cuda.empty_cache()
    print("Test passed!")

# TRAIN

test_before_train()
print(f"Model is on: {next(model.parameters()).device}")

# epoch level metrics
overall_val_losses = []
overall_train_losses = []

overall_train_bleu_scores = []
overall_val_bleu_scores = []

n_tqdm_cols = 100
use_tqdm = "tqdm" in sys.argv
tqdm_wrapper = lambda dl, desc: tqdm(dl, desc = desc, ncols = n_tqdm_cols, disable = use_tqdm)

for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    
    train_losses = []
    val_losses = []
    
    train_bleu_scores = []
    val_bleu_scores = []
    
    # train
    model.train()
    pbar = tqdm_wrapper(train_dl, " Train")

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

        loss, bleu_score = process(prediction, ans_ids)
        
        loss.backward()
        optimizer.step()        

        train_losses.append(loss.item())
        train_bleu_scores.append(bleu_score)
        
        current_train_loss = np.mean(train_losses)
        current_train_bleu_scores = np.mean(train_bleu_scores)
            
        pbar.set_postfix(loss = round(current_train_loss, 4), bleu = round(current_train_bleu_scores, 4))
        
    overall_train_losses.append(np.mean(train_losses))
    overall_train_bleu_scores.append(np.mean(train_bleu_scores))

	
    # val
    with torch.no_grad():
        pbar = tqdm_wrapper(val_dl, " Val  ")
        
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

            loss, bleu_score = process(prediction, ans_ids)
            
            val_losses.append(loss.item())
            val_bleu_scores.append(bleu_score)
            
            current_val_loss = np.mean(val_losses)
            current_val_bleu_scores = np.mean(val_bleu_scores)
            
            pbar.set_postfix(loss = round(current_val_loss, 4), bleu = round(current_val_bleu_scores, 4))
            
    overall_train_losses.append(np.mean(val_losses))
    overall_train_bleu_scores.append(np.mean(val_bleu_scores))
    
    scheduler.step(current_val_loss)
    
    if not use_tqdm:
        print(" Train loss:", round(current_train_loss, 4))
        print(" Val   loss:", round(current_val_loss, 4))
        print(" Train bleu:", round(current_train_bleu_scores, 4))
        print(" Val   bleu:", round(current_val_bleu_scores, 4))
    
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