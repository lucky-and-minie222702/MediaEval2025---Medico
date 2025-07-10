from ..custom_obj import *
from my_dataset import *
from torch.utils.data import DataLoader
import torch


# Tu nhien co tieng Trung
def invalid_char(texts):
    not_good = lambda x: sum([ord(c) > 255 for c in x]) > 0
    invalid_idx = [i for i, s in enumerate(texts) if not_good(s)]
    return invalid_idx

def drop_invalid_char_df(df):
    df.drop(index = invalid_char(df["question"].tolist()), inplace = True)
    df.drop(index = invalid_char(df["answer"].tolist()), inplace = True)


def load_and_save_data(question_max_length, answer_max_length):
    os.makedirs("data/save", exist_ok = True)
    
    # load df
    train_df = pd.read_csv("data/train.csv")
    drop_invalid_char_df(train_df)
    train_size = int(len(train_df) * 0.9)
    val_df = train_df.iloc[train_size::]
    train_df = train_df.iloc[:train_size:]
    test_df = pd.read_csv("data/test.csv")
    drop_invalid_char_df(test_df)


    print("Loading images")
    _ = get_img_dict("data/save/img_dict.joblib")

    tokenizer = MyText.MyTokenizer(
        1000,
        100,
    )
    
    print("Loading train data")
    train_ques_ids, train_ans_ids, tokenizer = get_ids(train_df, current_tokenizer = tokenizer, init_tokenizer = True, question_max_length = question_max_length, answer_max_length = answer_max_length)
    
    print("Loading test data")
    test_ques_ids, test_ans_ids = get_ids(test_df, current_tokenizer = tokenizer, init_tokenizer = False, question_max_length = question_max_length, answer_max_length = answer_max_length)
    
    print("Loading val data")
    val_ques_ids, val_ans_ids = get_ids(val_df, current_tokenizer = tokenizer, init_tokenizer = False, question_max_length = question_max_length, answer_max_length = answer_max_length)
    
    
    train_ques_ids = torch.tensor(train_ques_ids, dtype = torch.long)
    train_ans_ids = torch.tensor(train_ans_ids, dtype = torch.long)
    
    test_ques_ids = torch.tensor(test_ques_ids, dtype = torch.long)
    test_ans_ids = torch.tensor(test_ans_ids, dtype = torch.long)
    
    val_ques_ids = torch.tensor(val_ques_ids, dtype = torch.long)
    val_ans_ids = torch.tensor(val_ans_ids, dtype = torch.long)


    train_ds = MyDataset(train_df["img_id"].tolist(), train_ques_ids, train_ans_ids, transform = TRAIN_TRANSFORM)
    test_ds = MyDataset(test_df["img_id"].tolist(), test_ques_ids, test_ans_ids, transform = BASE_TRANSFORM)
    val_ds= MyDataset(val_df["img_id"].tolist(), val_ques_ids, val_ans_ids, transform = BASE_TRANSFORM)
    
    
    joblib.dump(train_ds, "data/save/train_ds.joblib")
    joblib.dump(test_ds, "data/save/test_ds.joblib")
    joblib.dump(val_ds, "data/save/val_ds.joblib")
    joblib.dump(tokenizer, "data/save/tokenizer.joblib")

    return train_ds, test_ds, val_ds, tokenizer


def load_saved_data(batch_size):
    """
    return train, test, val, tokenizer
    """
    
    img_dict = joblib.load("data/save/img_dict.joblib")
    
    
    train_ds: MyDataset = joblib.load("data/save/train_ds.joblib")
    test_ds: MyDataset = joblib.load("data/save/test_ds.joblib")
    val_ds: MyDataset = joblib.load("data/save/val_ds.joblib")
    
    train_ds.set_img_dict(img_dict)
    test_ds.set_img_dict(img_dict)
    val_ds.set_img_dict(img_dict)
    
    
    tokenizer: MyText.MyTokenizer = joblib.load("data/save/tokenizer.joblib")
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 4)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 4)
    
    return train_dl, test_dl, val_dl, tokenizer