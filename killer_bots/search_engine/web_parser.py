import article_parser

from killer_bots.search_engine.preprocess_docs import clean_wiki_text
from haystack.nodes import TransformersSummarizer
from haystack import Document
from summarizer import Summarizer

model = Summarizer()
# summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum")

url = "https://blog.unosquare.com/10-tips-for-writing-cleaner-code-in-any-programming-language"

title, content = article_parser.parse(
    url=url,
    output="markdown",
    timeout=5
)

# print(content)
content = clean_wiki_text(content)

docs = content.split("\n")
docs = [doc.strip() for doc in docs]
docs = [Document(doc) for doc in docs if len(doc) > 0]

print(len(docs))

summary = model(" ".join([doc.content for doc in docs]))
# summary = summarizer.predict(documents=docs, generate_single_summary=True)
print(summary)


# summary = summarizer.predict(documents=docs, generate_single_summary=False)
# print(summary)