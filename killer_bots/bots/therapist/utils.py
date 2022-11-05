import json
import os
import pathlib

import pandas as pd
import tqdm
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline
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


def get_document_store():
    try:
        os.remove("faiss_document_store.db")
    except:
        pass
    document_store = FAISSDocumentStore(
        sql_url="sqlite:///faiss_document_store.db",
        # embedding_dim=128,
        faiss_index_factory_str="Flat",
        return_embedding=True
    )
    return document_store


def get_documents():
    chats = get_chats()
    docs = [Document(chat["text"]) for chat in chats]
    return docs


def get_retriever(document_store):
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        model_format="sentence_transformers",
        scale_score=True
    )
    return retriever


def get_search_pipeline():
    document_store = get_document_store()
    docs = get_documents()
    document_store.write_documents(docs)
    retriever = get_retriever(document_store)
    document_store.update_embeddings(retriever)
    document_search_pipeline = DocumentSearchPipeline(retriever)
    return document_search_pipeline


if __name__ == "__main__":
    # chats = prepare_chats_from_db()
    # global_path = str(pathlib.Path(__file__).parent.resolve()) + "/database/prepared_chats.txt"
    # with open(global_path, 'w') as f:
    #     json.dump(chats, f)
    # print(len(chats))

    chats = get_chats()
    chats = [chat["text"] for chat in chats]
    print(len(chats))
