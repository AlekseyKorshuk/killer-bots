import time
from googlesearch import search
import article_parser
import requests
from transformers import AutoTokenizer

from killer_bots.search_engine.preprocess_docs import clean_wiki_text
from haystack.nodes import TransformersSummarizer
from haystack import Document
from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer

from fake_useragent import UserAgent
from bs4 import BeautifulSoup

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

    def __call__(self, query, num_results=1):
        links = self._get_links(query, num_results)
        summaries = []
        for i in range(num_results):
            link = next(links)
            print(link)
            content = self._get_article_text(link)
            summary = self._get_article_summary(content)
            summaries.append(summary)
        return summaries

    def _get_links(self, query, num_results):
        return search(query, num_results=num_results)

    def _get_article_text(self, url):
        ua = UserAgent()
        headers = {'User-Agent': str(ua.chrome)}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        soup = soup.find("body")
        para_text = soup.find_all(["p"])
        # content = "\n".join([result.text for result in para_text])
        return [result.text for result in para_text]

    def _get_article_summary(self, docs):
        docs = [doc.strip() for doc in docs]
        docs = [doc for doc in docs if len(doc) > 0]
        docs = [clean_wiki_text(doc) for doc in docs]
        body = "\n".join([doc for doc in docs])

        ratio = self._get_targen_summary_ratio(body)
        if ratio == 1:
            return body
        summary = ""
        for doc in docs:
            summary += self.summarizer(doc, ratio=ratio)
        return summary

    def _get_targen_summary_ratio(self, content):
        summary_tokens = self.tokenizer(content)["input_ids"]
        return min(self.target_num_tokens / len(summary_tokens), 1)


if __name__ == "__main__":
    search_engine = GoogleSearchEngine()
    summaries = search_engine("best way to learn python", num_results=1)
    print(summaries[0])