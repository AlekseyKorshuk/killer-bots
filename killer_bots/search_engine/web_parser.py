import time

import torch
from googlesearch import search
import article_parser
import requests
from transformers import AutoTokenizer

from killer_bots.search_engine.custom_pipeline import _get_document_store
from killer_bots.search_engine.preprocess_docs import clean_wiki_text, PreprocessDocs
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
        self.preprocessor = PreprocessDocs()
        self.query_retriever = SentenceTransformer("all-MiniLM-L6-v2")
        self.context_retriever = SentenceTransformer("all-MiniLM-L6-v2")
        self.retriever = EmbeddingRetriever(
            document_store=_get_document_store(),
            embedding_model="AlekseyKorshuk/retriever-coding-guru-adapted",
            model_format="sentence_transformers",
            scale_score=False
        )

    def __call__(self, query, num_results=1):
        links = self._get_links(query, num_results)
        summaries = []
        for i in range(num_results):
            link = next(links)
            print(link)
            content = self._get_article_text(link)
            docs = self._get_docs(content)
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
        content = "\n".join([doc.content for doc in needed_docs])
        return content

    def _get_links(self, query, num_results):
        return search(query, num_results=num_results)

    def _get_article_text(self, url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text
        # ua = UserAgent()
        # headers = {'User-Agent': str(ua.chrome)}
        # r = requests.get(url, headers=headers)
        # soup = BeautifulSoup(r.text, "html.parser")
        # soup = soup.find("body")
        # para_text = soup.find_all(["p"])
        # # content = "\n".join([result.text for result in para_text])
        # return [result.text for result in para_text]

    def _get_docs(self, content):
        docs = content.split("\n")
        docs = [doc.strip() for doc in docs]
        docs = [doc for doc in docs if len(doc) > 10]
        docs = [clean_wiki_text(doc) for doc in docs]
        docs = [Document(doc, meta={"name": None}) for doc in docs]
        # docs = self.preprocessor(docs)
        return docs

    def _get_article_summary(self, docs):

        body = "\n".join([doc.content for doc in docs])

        per = self._get_targen_summary_ratio(body)

        doc = self.nlp(body)
        tokens = [token.text for token in doc]
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in list(STOP_WORDS):
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        select_length = int(len(sentence_tokens) * per)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary = ''.join(final_summary)
        return summary

    def _get_targen_summary_ratio(self, content):
        summary_tokens = self.tokenizer(content)["input_ids"]
        return min(self.target_num_tokens / len(summary_tokens), 1)


if __name__ == "__main__":
    search_engine = GoogleSearchEngine()
    summaries = search_engine("best way to learn python", num_results=1)
    print(summaries[0])
