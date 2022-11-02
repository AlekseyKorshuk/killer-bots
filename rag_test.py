from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever

# document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

# Initialize DPR Retriever to encode documents, encode question and query documents
retriever = DensePassageRetriever(
    document_store=None,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True,
)

generator = RAGenerator(
    model_name_or_path="facebook/rag-sequence-nq",
    retriever=retriever,
    top_k=1,
    min_length=2
)