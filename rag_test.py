# import validators, re
# import torch
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
# import streamlit as st
# from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
# from sentence_transformers import SentenceTransformer
import spacy
# import time
# import base64
import requests
# import docx2txt
# from io import StringIO
# from PyPDF2 import PdfFileReader
import warnings
import nltk
import itertools
import numpy as np

nltk.download('punkt')

from nltk import sent_tokenize

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

def article_text_extractor(url: str):

    '''Extract text from url and divide text into chunks if length of text is more than 500 words'''

    ua = UserAgent()

    headers = {'User-Agent' :str(ua.chrome)}

    r = requests.get(url ,headers=headers)

    soup = BeautifulSoup(r.text, "html.parser")
    title_text = soup.find_all(["h1"])
    para_text = soup.find_all(["p"])
    article_text = [result.text for result in para_text]

    try:

        article_header = [result.text for result in title_text][0]

    except:

        article_header = ''

    article = nlp("\n".join(article_text))
    sentences = [i.text for i in list(article.sents)]

    current_chunk = 0
    chunks = []

    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= 500:
                chunks[current_chunk].extend(sentence.split(" "))
            else:
                current_chunk += 1
                chunks.append(sentence.split(" "))
        else:
            chunks.append(sentence.split(" "))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])

    return article_header, chunks, article


article_header, chunks, article = article_text_extractor("https://blog.unosquare.com/10-tips-for-writing-cleaner-code-in-any-programming-language")

print(len(chunks))

# for chunk in chunks:
#     print(chunk)
#     print()

print(article)