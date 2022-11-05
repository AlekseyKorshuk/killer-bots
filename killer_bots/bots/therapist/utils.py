import json
import pathlib

import pandas as pd
import tqdm
from summarizer import Summarizer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup


def prepare_chats_from_db():
    global_path = str(pathlib.Path(__file__).parent.resolve()) + "/database/counselchat-data.csv"
    df = pd.read_csv(global_path)
    summarizer = Summarizer()
    max_tokens = 128
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    chats = []
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        answer = row['answerText']
        soup = BeautifulSoup(answer, features="lxml")
        answer = soup.get_text()
        num_tokens = len(tokenizer(answer).input_ids)
        ratio = min(max_tokens / num_tokens, 1)
        if ratio != 0:
            answer = summarizer(answer, ratio=ratio)
        question_title = str(row['questionTitle']) if row['questionTitle'] else ""
        if question_title != "" and question_title[-1] not in [".", "!", "?"]:
            question_title = question_title + "."
        question_text = str(row['questionText']) if row['questionText'] else ""
        question = question_title + " " + question_text
        num_tokens = len(tokenizer(question).input_ids)
        ratio = min(max_tokens / num_tokens, 1)
        if ratio != 0:
            question = summarizer(question, ratio=ratio)
        chat = f"User: {question}\n" \
               f"Therapist: {answer}"
        chats.append(
            {"text": chat, "votes": row["upvotes"], "url": row["questionUrl"], "topics": row["topics"]}
        )
    return chats


def get_chats():
    global_path = str(pathlib.Path(__file__).parent.resolve()) + "/database/prepared_chats.txt"
    with open(global_path, "r") as f:
        chats = json.load(f)
    return chats


if __name__ == "__main__":
    # chats = prepare_chats_from_db()
    # global_path = str(pathlib.Path(__file__).parent.resolve()) + "/database/prepared_chats.txt"
    # with open(global_path, 'w') as f:
    #     json.dump(chats, f)
    # print(len(chats))

    chats = get_chats()
    print(len(chats))
