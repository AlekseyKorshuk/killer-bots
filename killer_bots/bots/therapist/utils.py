import pathlib

import pandas as pd
from summarizer import Summarizer
from transformers import AutoTokenizer


def get_chats_from_db():
    global_path = str(pathlib.Path(__file__).parent.resolve()) + "/database/counselchat-data.csv"
    df = pd.read_csv(global_path)
    summarizer = Summarizer()
    max_tokens = 128
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    chats = []
    for i, row in df.iterrows():
        answer = row['answerText']
        num_tokens = len(tokenizer(answer).input_ids)
        ratio = min(max_tokens / num_tokens, 1)
        if ratio != 0:
            answer = summarizer(answer, ratio=ratio)
        question_title = row['questionTitle'] if row['questionTitle'] else ""
        question_text = row['questionText'] if row['questionText'] else ""
        chats.append(
            f"User: {question_title + question_text}\n"
            f"Therapist: {answer}"
        )
    return chats


if __name__ == "__main__":
    chats = get_chats_from_db()
    print(len(chats))
