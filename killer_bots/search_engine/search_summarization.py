from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text, print_documents
from haystack.nodes import DensePassageRetriever, TransformersSummarizer
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
import os
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


def get_document_store():
    os.remove("faiss_document_store.db")
    document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")
    return document_store


def write_docs(document_store, doc_dir):
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)
    # Now, let's write the dicts containing documents to our DB.
    print(f"Number of docs: {len(docs)}")
    # import pdb; pdb.set_trace()
    document_store.write_documents(docs)
    return document_store


def get_retriever(doc_dir):
    document_store = get_document_store()
    document_store = write_docs(document_store, doc_dir)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )
    document_store.update_embeddings(retriever)
    return retriever


def test_retriever():
    retriever = get_retriever("/app/killer-bots/killer_bots/bots/code_guru/database")
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="What is SOLID?", params={"Retriever": {"top_k": 5}})
    print_documents(res)


def get_summarizer():
    summarizer = TransformersSummarizer(model_name_or_path="facebook/bart-large-cnn")
    return summarizer


def get_search_summarization_pipeline(doc_dir):
    summarizer = get_summarizer()
    retriever = get_retriever(doc_dir)
    pipeline = SearchSummarizationPipeline(summarizer=summarizer, retriever=retriever, return_in_answer_format=False)
    return pipeline


class SearchSummarization:
    params = {"Retriever": {"top_k": 5}, "Summarizer": {"generate_single_summary": True}}

    def __init__(self, doc_dir, params=None):
        self.pipeline = get_search_summarization_pipeline(doc_dir)
        if params is not None:
            self.params = params

    def __call__(self, query):
        res = self.pipeline.run(query=query, params=self.params)
        return res["documents"][0].content


def test_search_summarization_pipeline():
    pipeline = SearchSummarization("/app/killer-bots/killer_bots/bots/code_guru/database")
    query = "What is SOLID?"
    res = pipeline(query=query)
    print("Query:", query)
    print("Result:", res)


if __name__ == "__main__":
    test_retriever()
    test_search_summarization_pipeline()
