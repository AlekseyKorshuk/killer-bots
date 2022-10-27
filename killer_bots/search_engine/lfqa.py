import numpy as np
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text, print_documents
from haystack.nodes import DensePassageRetriever, TransformersSummarizer
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
from haystack import Document
from sklearn.metrics.pairwise import cosine_similarity

import os
import logging

from killer_bots.search_engine.utils import get_search_summarization_pipeline, get_retriever, get_summarizer
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


def get_lfqa_generator():
    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
    return generator


def get_lfqa_pipeline(doc_dir):
    generator = get_lfqa_generator()
    retriever = get_retriever(doc_dir)
    pipeline = GenerativeQAPipeline(generator, retriever)
    # import pdb; pdb.set_trace()
    return pipeline


class LFQA:
    params = {"Retriever": {"top_k": 5}, "Generator": {"top_k": 5, "do_sample": True, "max_length": 128}}

    def __init__(self, doc_dir, params=None):
        self.pipeline = get_lfqa_pipeline(doc_dir)
        if params is not None:
            self.params = params

    def __call__(self, query):
        res = self.pipeline.run(query=query, params=self.params)
        response = res["documents"][0].content.replace("\n", " ")
        if response[-1] != ".":
            response = response[:-1] + "."
        return response
        # return (response, cosine_score, dot_score)


def test_lfqa_pipeline():
    pipeline = LFQA("/app/killer-bots/killer_bots/bots/code_guru/database")
    while True:
        query = input("> ")
        print(pipeline(query))


TEST_QUESTIONS = [
    "What is SOLID?",
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
    test_lfqa_pipeline()
