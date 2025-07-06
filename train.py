from load_dataset import *
from models import *

train_dl, test_dl, val_dl, tokenizer = load_saved_data(batch_size = 16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyGRUModel(tokenizer.all_vocab_size + 1)

# TRAIN


get_padding_mask = lambda x: torch.clamp(x, max = 1)

with torch.no_grad():
    model.eval()
    for img, ques_ids, ans_ids in train_dl:
        out = model(
            image = img,
            questions = ques_ids,
            max_answer_length = 50,
            question_padding_masks = get_padding_mask(ques_ids),
            answers = ans_ids,
            teacher_forcing_ratio = 0.5,
        )  # (B, 50, vocab_size)
        example = torch.argmax(out[1, ::, ::], dim = -1).numpy()
        print(tokenizer.decode_sentence(example))
        break


# DONE

torch.cuda.empty_cache()