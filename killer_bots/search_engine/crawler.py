from haystack.nodes import Crawler

crawler = Crawler(output_dir="crawled_files")  # This tells the Crawler where to store the crawled files
docs = crawler.crawl(
    urls=[
        "https://www.digitalocean.com/community/conceptual-articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design",
    ],
    # This tells the Crawler which URLs to crawl
    crawler_depth=0  # This tells the Crawler to follow only the links that it finds on the initial URLs
)
