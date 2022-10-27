import numpy as np
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text, print_documents
from haystack.nodes import DensePassageRetriever, TransformersSummarizer
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
from haystack import Document
from sklearn.metrics.pairwise import cosine_similarity

import os
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


def get_document_store():
    os.remove("faiss_document_store.db")
    document_store = FAISSDocumentStore(embedding_dim=1024, faiss_index_factory_str="Flat")
    return document_store


def write_docs(document_store, doc_dir):
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)
    print(f"Number of docs: {len(docs)}")
    # import pdb; pdb.set_trace()
    document_store.write_documents(docs)
    return document_store


def get_retriever(doc_dir):
    document_store = get_document_store()
    document_store = write_docs(document_store, doc_dir)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="CarperAI/carptriever-1",
        passage_embedding_model="CarperAI/carptriever-1",
    )
    document_store.update_embeddings(retriever)
    return retriever


def test_retriever():
    retriever = get_retriever("/app/killer-bots/killer_bots/bots/code_guru/database")
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="What is SOLID?", params={"Retriever": {"top_k": 5}})
    print(res)
    print_documents(res)


def get_summarizer():
    summarizer = TransformersSummarizer(model_name_or_path="facebook/bart-large-cnn")
    return summarizer


def get_search_summarization_pipeline(doc_dir):
    summarizer = get_summarizer()
    retriever = get_retriever(doc_dir)
    pipeline = SearchSummarizationPipeline(summarizer=summarizer, retriever=retriever, return_in_answer_format=False)
    import pdb; pdb.set_trace()
    return pipeline


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
        return (response, cosine_score, dot_score)


def test_search_summarization_pipeline():
    pipeline = SearchSummarization("/app/killer-bots/killer_bots/bots/code_guru/database")
    while True:
        query = input("> ")
        print(pipeline(query))


if __name__ == "__main__":
    # test_retriever()
    test_search_summarization_pipeline()
