import time
from googlesearch import search
import article_parser
import requests

from killer_bots.search_engine.preprocess_docs import clean_wiki_text
from haystack.nodes import TransformersSummarizer
from haystack import Document
from summarizer import Summarizer
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

model = Summarizer()
# summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum")

url = "https://blog.unosquare.com/10-tips-for-writing-cleaner-code-in-any-programming-language"

# title, content = article_parser.parse(
#     url=url,
#     output="markdown",
#     timeout=5
# )

ua = UserAgent()

headers = {'User-Agent': str(ua.chrome)}

start = time.time()
r = requests.get(url, headers=headers)

soup = BeautifulSoup(r.text, "html.parser")

# input()
title_text = soup.find_all(["h1"])
para_text = soup.find_all(["p"])

content = "\n".join([result.text for result in para_text])
# print(content)
# content = clean_wiki_text(content)

docs = content.split("\n")
docs = [doc.strip() for doc in docs]
docs = [Document(doc) for doc in docs if len(doc) > 0]

# print(len(docs))

# print("\n".join([doc.content for doc in docs]))
# input()
body = "\n".join([doc.content for doc in docs])
optimal_k = model.calculate_optimal_k(body, k_max=20)
summary = model(body, num_sentences=optimal_k)
# summary = summarizer.predict(documents=docs, generate_single_summary=True)
print(summary)


# summary = summarizer.predict(documents=docs, generate_single_summary=False)
# print(summary)
print("Time taken:", time.time() - start)


class GoogleSearchEngine:
    def __init__(self):
        self.summarizer = Summarizer()
        self.target_num_tokens = 512


    def _get_links(self, query, num_results):
        return search(query, num_results=num_results)

