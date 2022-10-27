import numpy as np
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text, print_documents
from haystack.nodes import DensePassageRetriever, TransformersSummarizer
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
from haystack import Document
from sklearn.metrics.pairwise import cosine_similarity

import os
import logging

from killer_bots.search_engine.utils import get_search_summarization_pipeline, get_retriever

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


def test_retriever():
    retriever = get_retriever("/app/killer-bots/killer_bots/bots/code_guru/database")
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="What is SOLID?", params={"Retriever": {"top_k": 5}})
    # print(res)
    print_documents(res)


class SearchSummarization:
    params = {"Retriever": {"top_k": 5}, "Summarizer": {"generate_single_summary": True}}

    def __init__(self, doc_dir, params=None):
        self.pipeline = get_search_summarization_pipeline(doc_dir)
        if params is not None:
            self.params = params

    def __call__(self, query):
        res = self.pipeline.run(query=query, params=self.params)
        response = res["documents"][0].content
        query_embedding = self.pipeline.pipeline.graph._node['Retriever']['component'].embed_queries([query])
        document_embedding = self.pipeline.pipeline.graph._node['Retriever']['component'].embed_documents(
            [Document(response)]
        )
        # import pdb; pdb.set_trace()
        cosine_score = cosine_similarity(query_embedding, document_embedding)[0][0]
        dot_score = np.dot(query_embedding[0], document_embedding[0])
        return response
        # return (response, cosine_score, dot_score)


def test_search_summarization_pipeline():
    pipeline = SearchSummarization("/app/killer-bots/killer_bots/bots/code_guru/database")
    while True:
        query = input("> ")
        print(pipeline(query))


TEST_QUESTIONS = [
    "What is SOLID?",
    "What is the difference between a class and an object?",
    "What is Single Responsibility Principle?",
    "What is Open-Closed Principle?",
    "What is Liskov Substitution Principle?",
    "What is Interface Segregation Principle?",
    "What is Dependency Inversion Principle?",
    "Why should I use SOLID?",
    "Who created SOLID?",
    "How are you?",
    "What is your name?",
    "What is your age?",
    "What is your favorite color?",
    "How can you help me?",
]

if __name__ == "__main__":
    # test_retriever()
    test_search_summarization_pipeline()
