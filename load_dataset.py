from custom_obj import *
from dataset import *
from text_modules import *
from vision_modules import *
from modules import *
from torch.utils.data import DataLoader


def load_and_save_data():
    os.makedirs("data/save", exist_ok = True)
    
    # load df
    train_df = pd.read_csv("data/train.csv")
    train_size = int(len(train_df) * 0.8)
    val_df = train_df.iloc[train_size::]
    train_df = train_df.iloc[:train_size:]
    test_df = pd.read_csv("data/test.csv")


    print("Loading images")
    img_dict = get_img_dict("data/save/img_dict.joblib")

    tokenizer = MyText.MyTokenizer(
        1000,
        100,
    )
    
    print("Loading train data")
    train_ques_ids, train_ans_ids, tokenizer = get_ids(train_df, current_tokenizer = tokenizer, init_tokenizer = True, question_max_length = 35, answer_max_length = 50)
    
    print("\nLoading test data")
    test_ques_ids, test_ans_ids = get_ids(test_df, current_tokenizer = tokenizer, init_tokenizer = False, question_max_length = 35, answer_max_length = 50)
    
    print("Loading val data")
    val_ques_ids, val_ans_ids = get_ids(val_df, current_tokenizer = tokenizer, init_tokenizer = False, question_max_length = 35, answer_max_length = 50)


    train_ds = MyDataset(img_dict, train_df["img_id"].tolist(), train_ques_ids, train_ans_ids, transform = TRAIN_TRANSFORM)
    test_ds = MyDataset(img_dict, test_df["img_id"].tolist(), test_ques_ids, test_ans_ids, transform = BASE_TRANSFORM)
    val_ds= MyDataset(img_dict, val_df["img_id"].tolist(), val_ques_ids, val_ans_ids, transform = BASE_TRANSFORM)
    
    
    joblib.dump(train_ds, "data/save/train_ds.joblib")
    joblib.dump(test_ds, "data/save/test_ds.joblib")
    joblib.dump(val_ds, "data/save/val_ds.joblib")
    joblib.dump(tokenizer, "data/save/tokenizer.joblib")

    return train_ds, test_ds, val_ds, tokenizer


def load_saved_data(batch_size):
    train_ds = joblib.load("data/save/train_ds.joblib")
    test_ds = joblib.load("data/save/test_ds.joblib")
    val_ds = joblib.load("data/save/val_ds.joblib")
    tokenizer = joblib.load("data/save/tokenizer.joblib")
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
    
    return train_dl, test_dl, val_dl, tokenizer