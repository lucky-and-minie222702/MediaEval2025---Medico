from custom_obj import *
from dataset import *
from text_modules import *
from vision_modules import *
from modules import *
from torch.utils.data import DataLoader


def get_data_loader(batch_size):
    # load df
    train_df = pd.read_csv("data/train.csv")
    train_size = int(len(train_df) * 0.8)
    val_df = train_df.iloc[train_size::]
    train_df = train_df.iloc[:train_size:]
    test_df = pd.read_csv("data/test.csv")


    img_dict_save_path = "data/img_dict.joblib"
    if not path.exists(img_dict_save_path):
        img_dict = get_img_dict(img_dict_save_path)
    else:
        img_dict = joblib.load(img_dict_save_path)

    tokenizer = MyText.MyTokenizer(
        1000,
        100,
    )
    train_ques_ids, train_ans_ids, tokenizer = get_ids(train_df, current_tokenizer = tokenizer, init_tokenizer = True, question_max_length = 35, answer_max_length = 50)
    test_ques_ids, test_ans_ids = get_ids(test_df, current_tokenizer = tokenizer, init_tokenizer = False, question_max_length = 35, answer_max_length = 50)
    val_ques_ids, val_ans_ids = get_ids(val_df, current_tokenizer = tokenizer, init_tokenizer = False, question_max_length = 35, answer_max_length = 50)


    train_dataset = MyDataset(img_dict, train_df["img_id"].tolist(), train_ques_ids, train_ans_ids, transform = TRAIN_TRANSFORM)
    test_dataset = MyDataset(img_dict, test_df["img_id"].tolist(), test_ques_ids, test_ans_ids, transform = BASE_TRANSFORM)
    val_dataset = MyDataset(img_dict, val_df["img_id"].tolist(), val_ques_ids, val_ans_ids, transform = BASE_TRANSFORM)


    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)


    return train_dl, test_dl, val_dl