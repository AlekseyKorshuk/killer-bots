import article_parser

from killer_bots.search_engine.preprocess_docs import clean_wiki_text

url = "https://blog.unosquare.com/10-tips-for-writing-cleaner-code-in-any-programming-language"

title, content = article_parser.parse(
    url=url,
    output="markwodn",
    timeout=5
)

content = clean_wiki_text(content)

docs = content.split("\n")
docs = [clean_wiki_text(doc.strip()) for doc in docs]
docs = [doc for doc in docs if len(doc) > 0]

print(len(docs))