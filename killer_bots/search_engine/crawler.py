from haystack import Document
from haystack.nodes import Crawler, PreProcessor

crawler = Crawler(output_dir="refactoring_guru")  # This tells the Crawler where to store the crawled files
docs = crawler.crawl(
    urls=[
        "https://refactoring.guru/design-patterns/what-is-pattern",
    ],
    # This tells the Crawler which URLs to crawl
    crawler_depth=0  # This tells the Crawler to follow only the links that it finds on the initial URLs
)
docs = [Document(doc) for doc in docs]
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
)
docs = preprocessor.process(docs)
import pdb; pdb.set_trace()
