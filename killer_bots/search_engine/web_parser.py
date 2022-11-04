import re
import time
from functools import wraps

import markdownify
import torch
import tqdm
from googlesearch import search
import article_parser
import requests
from summarizer.sbert import SBertSummarizer
from transformers import AutoTokenizer

from killer_bots.search_engine.custom_pipeline import _get_document_store
from killer_bots.search_engine.preprocess_docs import clean_wiki_text, PreprocessDocs, PreprocessDocsFast, join_docs
from haystack.nodes import TransformersSummarizer, EmbeddingRetriever
from haystack import Document
from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer
from newspaper import Article
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from string import punctuation
from heapq import nlargest
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
from sklearn.decomposition import PCA

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

def get_markdown_website(url):
    ua = UserAgent()
    headers = {'User-Agent': str(ua.chrome)}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    soup = soup.find("body")
    tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "code", "pre"]
    soup = soup.find_all(tags, recursive=True)
    html = "\n".join([str(result) for result in soup])
    text = markdownify.markdownify(html, heading_style="ATX", wrap=True, wrap_width=10**6)
    text = re.sub(r"!\[(.*?)\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub("[\[].*?[\]]", "", text)
    return text


class GoogleSearchEngine:
    def __init__(self):
        self.summarizer = Summarizer()
        self.summarizer = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        self.target_num_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
        self.preprocessor = PreprocessDocsFast()
        self.query_retriever = SentenceTransformer("all-MiniLM-L6-v2")
        self.context_retriever = SentenceTransformer("all-MiniLM-L6-v2")

    @timeit
    def __call__(self, query, top_k=1):
        links = self._get_links(query, top_k)
        summaries = []
        for i in range(top_k):
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
        _, ids = torch.topk(cosine_scores[0], len(docs))
        needed_docs = [docs[ids[0]]]
        for id in ids[1:]:
            context = "\n".join([doc.content for doc in needed_docs])
            num_tokens = len(self.tokenizer(context)["input_ids"])
            new_doc_tokens = len(self.tokenizer(docs[id].content)["input_ids"])
            if num_tokens + new_doc_tokens < self.target_num_tokens:
                needed_docs.append(docs[id])
            else:
                break
        needed_docs = sorted(needed_docs, key=lambda x: x.meta["id"], reverse=False)
        content = "\n".join([doc.content for doc in needed_docs])
        if len(self.tokenizer(content)["input_ids"]) > self.target_num_tokens:
            content = self._get_article_summary(needed_docs)
        return content

    def _get_links(self, query, num_results):
        return search(query, num_results=num_results)

    def _get_article_text(self, url):
        text = get_markdown_website(url)
        return text
        # article = Article(url)
        # article.download()
        # article.parse()
        # return article.text, article.html

    def _get_docs(self, content):
        docs = content.split("\n")
        docs = [doc.strip() for doc in docs]
        docs = [doc for doc in docs if len(doc) > 0]
        # docs = [clean_wiki_text(doc) for doc in docs]
        docs = [Document(doc, meta={"name": None}) for doc in docs]
        print("From:", len(docs))
        docs = self.preprocessor(docs)
        docs = [doc for doc in docs if len(doc.content) > 30]
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


class GoogleSearchEngine2(GoogleSearchEngine):
    def __init__(self):
        super().__init__()
        self.model = SBertSummarizer('msmarco-distilbert-base-tas-b')

    def _get_needed_content(self, query, docs):
        data = [doc.content for doc in docs]
        # data = []
        # for doc in tqdm.tqdm(docs):
        #     num_sentences = self.model.calculate_optimal_k(doc.content)
        #     summary = self.model(doc.content, num_sentences=num_sentences)
        #     data.append(summary)

        sentences, embeddings = self.model.cluster_runner(
            sentences=data,
            ratio=1.0,
            num_sentences=len(data),
        )

        pca_model = PCA(n_components=2)
        pca_components = pca_model.fit_transform(embeddings)

        cluster_model = AffinityPropagation(damping=0.75)
        cluster_model.fit(pca_components)
        yhat = cluster_model.predict(pca_components)

        groups = {x: [] for x in unique(yhat)}
        for i, cluster in enumerate(yhat):
            groups[cluster].append(i)
        groups = {k: v for k, v in groups.items() if len(v) > 1}

        grouped_docs = []
        for cluster in groups.values():
            grouped_docs.append(
                join_docs([docs[i] for i in cluster])
            )
        docs = grouped_docs
        data = [doc.content for doc in docs]
        data = [query] + data
        sentences, embeddings = self.model.cluster_runner(
            sentences=data,
            ratio=1.0,
            num_sentences=len(data),
        )

        pca_model = PCA(n_components=2)
        pca_components = pca_model.fit_transform(embeddings)
        distances = util.dot_score(pca_components, pca_components)[0][1:]
        # distances = euclidean_distances(pca_components, pca_components)
        # distances = distances[0][1:]
        print(distances)
        print([sentence.split("\n")[0] for sentence in sentences])
        _, ids = torch.topk(distances, len(docs))
        needed_docs = [docs[ids[0]]]
        for id in ids[1:]:
            context = "\n".join([doc.content for doc in needed_docs])
            num_tokens = len(self.tokenizer(context)["input_ids"])
            new_doc_tokens = len(self.tokenizer(docs[id].content)["input_ids"])
            if num_tokens + new_doc_tokens < self.target_num_tokens:
                needed_docs.append(docs[id])
            else:
                break
        needed_docs = sorted(needed_docs, key=lambda x: x.meta["id"], reverse=False)
        content = "\n".join([doc.content for doc in needed_docs])
        if len(self.tokenizer(content)["input_ids"]) > self.target_num_tokens:
            content = self._get_article_summary(needed_docs)
        return content


if __name__ == "__main__":
    search_engine = GoogleSearchEngine2()
    summaries = search_engine("coding, types of design patterns?", top_k=1)
    for summary in summaries:
        print("#" * 100)
        print(summary)
