from haystack.nodes import Crawler

if __name__ == "__main__":
    crawler = Crawler(output_dir="crawled_files")  # This tells the Crawler where to store the crawled files
    docs = crawler.crawl(
        urls=["https://haystack.deepset.ai/docs/get-started"],  # This tells the Crawler which URLs to crawl
        filter_urls=["haystack"],  # Here, you can pass regular expressions that the crawled URLs must comply with
        crawler_depth=1  # This tells the Crawler to follow only the links that it finds on the initial URLs
    )
