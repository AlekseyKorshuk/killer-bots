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
import article_parser
import re

nltk.download('punkt')

from nltk import sent_tokenize

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_lg")


def article_text_extractor(url: str):
    '''Extract text from url and divide text into chunks if length of text is more than 500 words'''

    # ua = UserAgent()
    #
    # headers = {'User-Agent': str(ua.chrome)}
    #
    # r = requests.get(url, headers=headers)
    #
    # soup = BeautifulSoup(r.text, "html.parser")
    # title_text = soup.find_all(["h1"])
    # para_text = soup.find_all(["p"])
    # article_text = [result.text for result in para_text]

    article_header, article_text = article_parser.parse(
        url=url,
        output='markdown',
        timeout=5
    )
    article_text = re.sub('s/\!\{0,1\}\[[^]]*\]([^)]*)//g', '', article_text)
    article_text = re.sub('s/!\?\[.*\](.*)//g', '', article_text)

    article = nlp(article_text)
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

    return article_header, chunks, article_text


article_header, chunks, article = article_text_extractor(
    "https://blog.unosquare.com/10-tips-for-writing-cleaner-code-in-any-programming-language")

print(len(chunks))

for chunk in chunks:
    print(chunk)
    print()
print("#" * 100)
print(article)
# print("#" * 100)
#
# article = nlp(content)
#
# print(title)
# print(article)
