from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text
from haystack.nodes import DensePassageRetriever
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline


def get_document_store():
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)
    return document_store


def write_docs(document_store, doc_dir):
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    # Now, let's write the dicts containing documents to our DB.
    print(f"Number of docs: {len(docs)}")
    document_store.write_documents(docs)
    return document_store


def get_retriever(doc_dir):
    document_store = get_document_store()
    write_docs(document_store, doc_dir)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )
    document_store.update_embeddings(retriever)
    return retriever


def test_retriever():
    retriever = get_retriever("../bots/code_guru/database")
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="What is SOLID?", params={"Retriever": {"top_k": 3}})
    print_documents(res, max_text_len=512)


if __name__ == "__main__":
    test_retriever()
