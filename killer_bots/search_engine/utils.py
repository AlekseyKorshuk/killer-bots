import os
from haystack.document_stores import FAISSDocumentStore
import numpy as np
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text, print_documents
from haystack.nodes import DensePassageRetriever, TransformersSummarizer, Seq2SeqGenerator, PreProcessor, RAGenerator, \
    BM25Retriever, SentenceTransformersRanker, EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline, GenerativeQAPipeline
from haystack import Document, Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request

import os
import logging

from transformers import AutoTokenizer


def get_document_store():
    os.remove("faiss_document_store.db")
    document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat", return_embedding=True)
    return document_store


def write_docs(document_store, doc_dir):
    change_extentions_to_txt(doc_dir)
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(docs)
    # docs = get_huggingface_course_docs()

    print(f"Number of docs: {len(docs)}")
    # import pdb; pdb.set_trace()
    document_store.write_documents(docs)
    return document_store


def get_retriever(doc_dir):
    document_store = get_document_store()
    document_store = write_docs(document_store, doc_dir)
    # retriever = DensePassageRetriever(
    #     document_store=document_store,
    #     query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    #     passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    #     use_gpu=True,
    #     embed_title=True,
    # )
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )
    # retriever = BM25Retriever(document_store=document_store)
    document_store.update_embeddings(retriever)
    return retriever


def get_summarizer():
    summarizer = TransformersSummarizer(model_name_or_path="facebook/bart-large-cnn")
    return summarizer


def get_search_summarization_pipeline(doc_dir):
    summarizer = get_summarizer()
    retriever = get_retriever(doc_dir)
    pipeline = SearchSummarizationPipeline(summarizer=summarizer, retriever=retriever, return_in_answer_format=False)
    # import pdb; pdb.set_trace()
    return pipeline


def get_lfqa_generator():
    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
    return generator


def get_rag_generator():
    generator = RAGenerator(
        model_name_or_path="facebook/rag-token-nq",
        use_gpu=True,
        top_k=1,
        max_length=300,
        min_length=50,
        embed_title=True,
        num_beams=2,
    )
    # generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
    return generator


def get_lfqa_pipeline(doc_dir):
    generator = get_lfqa_generator()
    # generator = get_rag_generator()
    retriever = get_retriever(doc_dir)
    pipeline = GenerativeQAPipeline(generator, retriever)
    return pipeline


def change_extentions_to_txt(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".md"):
                os.rename(os.path.join(root, file), os.path.join(root, file.replace(".md", ".txt")))


def get_huggingface_course_docs():
    doc_dir = "huggingface_course"
    url = "https://github.com/AlekseyKorshuk/cdn/raw/main/huggingface-course/huggingface%20course%20main%20chapters-en.zip"
    # urllib.request.urlretrieve(
    #     "https://downgit.github.io/#/home?url=https://github.com/huggingface/course/tree/main/chapters/en",
    #     "huggingface_course.zip")
    if not os.path.exists(doc_dir):
        fetch_archive_from_http(url=url, output_dir=doc_dir)
    change_extentions_to_txt(doc_dir)
    docs = convert_files_to_docs(dir_path=doc_dir, split_paragraphs=True)
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


def test():
    doc_dir = "/app/killer-bots/killer_bots/bots/code_guru/database"
    try:
        os.remove("faiss_full_document_store.db")
    except:
        pass
    document_store = FAISSDocumentStore(
        sql_url="sqlite:///faiss_full_document_store.db",
        # embedding_dim=128,
        faiss_index_factory_str="Flat",
        return_embedding=True
    )
    change_extentions_to_txt(doc_dir)
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=False)

    # import pdb; pdb.set_trace()
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(docs)
    document_store.write_documents(docs)

    num_docs = len(docs)
    # num_docs = 3
    print(f"Number of docs: {num_docs}")

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers"
    )
    document_store.update_embeddings(retriever)
    p_retrieval = DocumentSearchPipeline(retriever)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    while True:
        query = input("> ")
        # query = "why i need to refactor my code?"
        res = p_retrieval.run(query=query, params={"Retriever": {"top_k": num_docs}})
        documents = res["documents"]
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
        print(stats)
        top_files = list(stats.keys())[:5]
        top_docs = []
        ids = []
        for i, doc in enumerate(documents):
            if doc.meta["name"] in top_files:
                top_docs.append(doc)
                ids.append(i)
        top_docs = top_docs[:5]
        ids = ids[:5]

        top_docs = sorted(top_docs, key=lambda x: x.meta['vector_id'], reverse=False)
        top_docs = sorted(top_docs, key=lambda x: x.meta['name'], reverse=False)
        scores = [doc.score for doc in top_docs]
        print_documents({"documents": top_docs, "query": query}, max_text_len=None)
        print(scores)
        joined_text = "\n".join([doc.content for doc in top_docs])
        num_tokens = len(tokenizer(joined_text).input_ids)
        print(f"num_tokens: {num_tokens}")
        num_tokens = len(tokenizer(joined_text).input_ids[0])
        print(f"num_tokens: {num_tokens}")


if __name__ == "__main__":
    test()
