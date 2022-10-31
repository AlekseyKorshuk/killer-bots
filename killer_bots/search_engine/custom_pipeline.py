import os

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import convert_files_to_docs
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from killer_bots.search_engine.utils import change_extentions_to_txt

import re


def clean_wiki_text(text: str) -> str:
    # get rid of multiple new lines
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    # remove extremely short lines
    lines = text.split("\n")
    cleaned = []
    for l in lines:
        if len(l) > 30:
            cleaned.append(l)
        elif l[:2] == "==" and l[-2:] == "==":
            cleaned.append(l)
    text = "\n".join(cleaned)

    # add paragraphs (identified by wiki section title which is always in format "==Some Title==")
    text = text.replace("\n==", "\n\n\n==")

    # remove empty paragrahps
    text = re.sub(r"(==.*==\n\n\n)", "", text)

    # remove multiple dashes
    # text = re.sub(r"#+ +", "", text)
    # text = re.sub(r" +#+", "", text)

    return text


def _get_document_store():
    try:
        os.remove("faiss_document_store.db")
    except:
        pass
    document_store = FAISSDocumentStore(
        sql_url="sqlite:///faiss_document_store.db",
        # embedding_dim=128,
        faiss_index_factory_str="Flat",
        return_embedding=True
    )
    return document_store


def get_documents(doc_dir):
    change_extentions_to_txt(doc_dir)
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=False)

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(docs)
    return docs


def get_retriever(document_store):
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers"
    )
    return retriever


def get_file_scores(documents):
    stats = {}
    counter = {}
    for doc in documents:
        if doc.meta["name"] not in stats:
            stats[doc.meta["name"]] = 0
            counter[doc.meta["name"]] = 0
        stats[doc.meta["name"]] += doc.score
        counter[doc.meta["name"]] += 1
    for key in stats:
        stats[key] /= counter[key]
    # sort by score descending order
    stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}
    return stats


def get_top_docs(documents, stats, top_k=5):
    top_files = list(stats.keys())[:top_k]
    top_docs = []
    for doc in documents:
        if doc.meta["name"] in top_files:
            top_docs.append(doc)
    top_docs = top_docs[:top_k]
    return top_docs


def postprocess_top_docs(top_docs):
    docs_match = {}
    top_docs = sorted(top_docs, key=lambda x: (x.score, x.meta['vector_id']), reverse=True)
    # top_docs = sorted(top_docs, key=lambda x: x.meta['vector_id'], reverse=False)
    # top_docs = sorted(top_docs, key=lambda x: x.meta['name'], reverse=False)
    return top_docs


def postprocess_answer(top_docs):
    joined_text = "\n".join([doc.content for doc in top_docs])
    return joined_text


class Pipeline:
    def __init__(self, doc_dir):
        self.document_store = _get_document_store()
        docs = get_documents(doc_dir)
        self.num_docs = len(docs)
        self.document_store.write_documents(docs)
        self.retriever = get_retriever(self.document_store)
        self.document_store.update_embeddings(self.retriever)
        self.document_search_pipeline = DocumentSearchPipeline(self.retriever)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")

    def get_relevant_docs(self, query):
        documents = self.document_search_pipeline.run(
            query=query,
            params={
                "Retriever": {
                    "top_k": self.num_docs
                }
            }
        )["documents"]
        return documents

    def print_logs(self, query, top_docs, answer):
        print("Query:", query)
        print("Docs:")
        for doc in top_docs:
            print(doc.meta["vector_id"], doc.meta["name"], doc.score)
        num_tokens = len(self.tokenizer(answer).input_ids)
        print(f"Number of tokens: {num_tokens}")

    def __call__(self, query, top_k=5, verbose=False):
        documents = self.get_relevant_docs(query)
        stats = get_file_scores(documents)
        top_docs = get_top_docs(documents, stats, top_k)
        top_docs = postprocess_top_docs(top_docs)
        answer = postprocess_answer(top_docs)
        if verbose:
            self.print_logs(query, top_docs, answer)
        return answer


if __name__ == "__main__":
    pipeline = Pipeline("/app/killer-bots/killer_bots/bots/code_guru/database")
    while True:
        query = input("> ")
        print(pipeline(query, verbose=True))
