import logging
import os

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.label_generator import PseudoLabelGenerator


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# We load the TAS-B model, a state-of-the-art model trained on MS MARCO
max_seq_length = 200
model_name = "msmarco-distilbert-base-tas-b"

org_model = SentenceTransformer(model_name)
org_model.max_seq_length = max_seq_length

# We define a simple query and some documents how diseases are transmitted
# As TAS-B was trained on rather out-dated data (2018 and older), it has now idea about COVID-19
# So in the below example, it fails to recognize the relationship between COVID-19 and Corona


def show_examples(model):
    query = "How is COVID-19 transmitted"
    docs = [
        "Corona is transmitted via the air",
        "Ebola is transmitted via direct contact with blood",
        "HIV is transmitted via sex or sharing needles",
        "Polio is transmitted via contaminated water or food",
    ]

    query_emb = model.encode(query)
    docs_emb = model.encode(docs)
    scores = util.dot_score(query_emb, docs_emb)[0]
    doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print("Query:", query)
    for doc, score in doc_scores:
        # print(doc, score)
        print(f"{score:0.02f}\t{doc}")


print("Original Model")
show_examples(org_model)

dataset = load_dataset("nreimers/trec-covid", split="train")
num_documents = 10000
corpus = []
for row in dataset:
    if len(row["title"]) > 20 and len(row["text"]) > 100:
        text = row["title"] + " " + row["text"]

        text_lower = text.lower()

        # The dataset also contains many papers on other diseases. To make the training in this demo
        # more efficient, we focus on papers that talk about COVID.
        if "covid" in text_lower or "corona" in text_lower or "sars-cov-2" in text_lower:
            corpus.append(text)

        if len(corpus) >= num_documents:
            break

print("Len Corpus:", len(corpus))

try:
    os.remove("faiss_document_store_gpl.db")
except:
    pass

document_store = FAISSDocumentStore(
    sql_url="sqlite:///faiss_document_store_gpl.db",
    faiss_index_factory_str="Flat",
    similarity="cosine"
)
document_store.write_documents([{"content": t} for t in corpus])


retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b",
    model_format="sentence_transformers",
    max_seq_len=max_seq_length,
    progress_bar=False,
)
document_store.update_embeddings(retriever)



use_question_generator = True


if use_question_generator:
    questions_producer = QuestionGenerator(
        model_name_or_path="doc2query/msmarco-t5-base-v1",
        max_length=64,
        split_length=128,
        batch_size=32,
        num_queries_per_doc=3,
    )

else:
    questions_producer = query_doc_pairs

# We can use either QuestionGenerator or already generated questions in PseudoLabelGenerator
psg = PseudoLabelGenerator(questions_producer, retriever, max_questions_per_document=10, batch_size=32, top_k=10)
output, pipe_id = psg.run(documents=document_store.get_all_documents())

retriever.train(output["gpl_labels"])

print("Original Model")
show_examples(org_model)

print("\n\nAdapted Model")
show_examples(retriever.embedding_encoder.embedding_model)

retriever.save("adapted_retriever")
