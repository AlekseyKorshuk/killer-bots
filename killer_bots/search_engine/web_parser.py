import re
import time
from functools import wraps

import torch
from googlesearch import search
import article_parser
import requests
from transformers import AutoTokenizer

from killer_bots.search_engine.custom_pipeline import _get_document_store
from killer_bots.search_engine.preprocess_docs import clean_wiki_text, PreprocessDocs, PreprocessDocsFast
from haystack.nodes import TransformersSummarizer, EmbeddingRetriever
from haystack import Document
from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer
from newspaper import Article
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sentence_transformers import util, SentenceTransformer


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

# model = Summarizer()
# # summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum")
#
# url = "https://blog.unosquare.com/10-tips-for-writing-cleaner-code-in-any-programming-language"
#
# # title, content = article_parser.parse(
# #     url=url,
# #     output="markdown",
# #     timeout=5
# # )
#
# ua = UserAgent()
#
# headers = {'User-Agent': str(ua.chrome)}
#
# start = time.time()
# r = requests.get(url, headers=headers)
#
# soup = BeautifulSoup(r.text, "html.parser")
#
# # input()
# title_text = soup.find_all(["h1"])
# para_text = soup.find_all(["p"])
#
# content = "\n".join([result.text for result in para_text])
# # print(content)
# # content = clean_wiki_text(content)
#
# docs = content.split("\n")
# docs = [doc.strip() for doc in docs]
# docs = [Document(doc) for doc in docs if len(doc) > 0]
#
# # print(len(docs))
#
# # print("\n".join([doc.content for doc in docs]))
# # input()
# body = "\n".join([doc.content for doc in docs])
# res = model.calculate_optimal_k(body, k_max=20)
# summary = model(body)
# # summary = summarizer.predict(documents=docs, generate_single_summary=True)
# print(summary)
#
# # summary = summarizer.predict(documents=docs, generate_single_summary=False)
# # print(summary)
# print("Time taken:", time.time() - start)


class GoogleSearchEngine:
    def __init__(self):
        # self.summarizer = Summarizer()
        self.summarizer = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        self.target_num_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
        self.nlp = spacy.load('en_core_web_sm')
        self.preprocessor = PreprocessDocsFast()
        self.query_retriever = SentenceTransformer("all-MiniLM-L6-v2")
        self.context_retriever = SentenceTransformer("all-MiniLM-L6-v2")

    @timeit
    def __call__(self, query, num_results=1):
        links = self._get_links(query, num_results)
        summaries = []
        for i in range(num_results):
            link = next(links)
            print(link)
            content, html = self._get_article_text(link)
            docs = self._get_docs(content, html)
            summary = self._get_needed_content(query, docs)
            # summary = self._get_article_summary(content)
            summaries.append(summary)
        return summaries

    def _get_needed_content(self, query, docs):
        # print(docs)
        query_embeddings = self.query_retriever.encode([query], convert_to_tensor=True)
        context_embeddings = self.context_retriever.encode([doc.content for doc in docs], convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embeddings, context_embeddings)
        # print(cosine_scores)
        top_k = 5
        _, ids = torch.topk(cosine_scores[0], top_k)
        needed_docs = [docs[i] for i in ids]
        # content = "\n".join([doc.content for doc in needed_docs])
        content = self._get_article_summary(needed_docs)
        return content

    def _get_links(self, query, num_results):
        return search(query, num_results=num_results)

    def _get_article_text(self, url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text, article.html
        # ua = UserAgent()
        # headers = {'User-Agent': str(ua.chrome)}
        # r = requests.get(url, headers=headers)
        # soup = BeautifulSoup(r.text, "html.parser")
        # soup = soup.find("body")
        # para_text = soup.find_all(["p"])
        # # content = "\n".join([result.text for result in para_text])
        # return [result.text for result in para_text]

    def _get_docs(self, content, html):
        docs = content.split("\n")
        docs = [doc.strip() for doc in docs]
        docs = [doc for doc in docs if len(doc) > 30]
        docs = [clean_wiki_text(doc) for doc in docs]
        docs = [Document(doc, meta={"name": None}) for doc in docs]
        print("From:", len(docs))
        soup = BeautifulSoup(html, "html.parser")
        headers = soup.find_all(re.compile('^h[1-6]$'))
        headers = [header.text for header in headers]
        headers = [clean_wiki_text(header) for header in headers]
        docs = self.preprocessor(docs, headers)
        print("To:", len(docs))
        return docs

    def _get_article_summary(self, docs):

        body = "\n".join([doc.content for doc in docs])

        per = self._get_targen_summary_ratio(body)
        print("Ratio:", per)
        summary = self.summarizer(body, ratio=per)
        return summary

    def _get_targen_summary_ratio(self, content):
        summary_tokens = self.tokenizer(content)["input_ids"]
        return min(self.target_num_tokens / len(summary_tokens), 1)


if __name__ == "__main__":
    search_engine = GoogleSearchEngine()
    summaries = search_engine("best way to learn python", num_results=3)
    for summary in summaries:
        print("#" * 100)
        print(summary)
