import numpy as np
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text, print_documents
from haystack.nodes import DensePassageRetriever, TransformersSummarizer
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
from haystack import Document
from sklearn.metrics.pairwise import cosine_similarity

import os
import logging

from killer_bots.search_engine.utils import get_search_summarization_pipeline, get_retriever, get_summarizer, \
    get_lfqa_pipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


class LFQA:
    params = {"Retriever": {"top_k": 5}, "Generator": {"top_k": 5}}

    def __init__(self, doc_dir, params=None):
        self.pipeline = get_lfqa_pipeline(doc_dir)
        if params is not None:
            self.params = params

    def __call__(self, query):
        res = self.pipeline.run(query=query, params=self.params)
        # print(res)
        # response = res["answers"][0].answer
        # print("Answer:", res["answers"][0].answer)

        response = res["documents"][0].content
        if response[-1] != ".":
            response = response[:-1] + "."
        return response
        # return (response, cosine_score, dot_score)


def test_lfqa_pipeline():
    pipeline = LFQA("/app/killer-bots/killer_bots/bots/code_guru/database")
    while True:
        query = input("> ")
        print(pipeline(query))


def evaluate_lfqa_pipeline(questions):
    pipeline = LFQA("/app/killer-bots/killer_bots/bots/code_guru/database")
    for question in questions:
        print()
        print("Question:", question)
        print("Answer:", pipeline(question))


TEST_QUESTIONS = [
    "What is SOLID?",
    "What is Single Responsibility Principle?",
    "What is Open-Closed Principle?",
    "What is Liskov Substitution Principle?",
    "What is Interface Segregation Principle?",
    "What is Dependency Inversion Principle?",
    "Why should I use SOLID?",
    "Who created SOLID?",
    "Please name all SOLID principles?",
]

if __name__ == "__main__":
    # evaluate_lfqa_pipeline(TEST_QUESTIONS)
    test_lfqa_pipeline()
