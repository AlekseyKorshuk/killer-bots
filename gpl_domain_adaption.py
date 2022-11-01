import logging
import os

from haystack.utils import convert_files_to_docs
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.label_generator import PseudoLabelGenerator

from killer_bots.search_engine.preprocess_docs import preprocess_docs, clean_wiki_text
#
# logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)
#
# # We load the TAS-B model, a state-of-the-art model trained on MS MARCO
# model_name = "multi-qa-mpnet-base-dot-v1"
#
# org_model = SentenceTransformer(model_name)
#
#
# # We define a simple query and some documents how diseases are transmitted
# # As TAS-B was trained on rather out-dated data (2018 and older), it has now idea about COVID-19
# # So in the below example, it fails to recognize the relationship between COVID-19 and Corona
#
#
# def show_examples(model):
#     query = "what is technical debt?"
#     docs = [
#         'Causes of technical debt\nBusiness pressure\nSometimes business circumstances might force you to roll out features before they’re completely finished. In this case, patches and kludges will appear in the code to hide the unfinished parts of the project.\nLack of understanding of the consequences of technical debt\nSometimes your employer might not understand that technical debt has “interest” insofar as it slows down the pace of development as debt accumulates. This can make it too difficult to dedicate the team’s time to refactoring because management doesn’t see the value of it.',
#         'Everyone does their best to write excellent code from scratch. There probably isn’t a programmer out there who intentionally writes unclean code to the detriment of the project. But at what point does clean code become unclean?\nThe metaphor of “technical debt” in regards to unclean code was originally suggested by Ward Cunningham.'  + 'If you get a loan from a bank, this allows you to make purchases faster. You pay extra for expediting the process - you don’t just pay off the principal, but also the additional interest on the loan. Needless to say, you can even rack up so much interest that the amount of interest exceeds your total income, making full repayment impossible.\nThe same thing can happen with code. You can temporarily speed up without writing tests for new features, but this will gradually slow your progress every day until you eventually pay off the debt by writing tests.',
#         'Lack of compliance monitoring\nThis happens when everyone working on the project writes code as they see fit (i.e. the same way they wrote the last project).\nIncompetence\nThis is when the developer just doesn’t know how to write decent code.',
#         'Failing to combat the strict coherence of components\nThis is when the project resembles a monolith rather than the product of individual modules. In this case, any changes to one part of the project will affect others. Team development is made more difficult because it’s difficult to isolate the work of individual members.\nLack of tests\nThe lack of immediate feedback encourages quick, but risky workarounds or kludges. In worst cases, these changes are implemented and deployed right into the production without any prior testing. The consequences can be catastrophic. For example, an innocent-looking hotfix might send a weird test email to thousands of customers or even worse, flush or corrupt an entire database.\nLack of documentation\nThis slows down the introduction of new people to the project and can grind development to a halt if key people leave the project.\nLack of interaction between team members\nIf the knowledge base isn’t distributed throughout the company, people will end up working with an outdated understanding of processes and information about the project. This situation can be exacerbated when junior developers are incorrectly trained by their mentors.\nLong-term simultaneous development in several branches\nThis can lead to the accumulation of technical debt, which is then increased when changes are merged. The more changes made in isolation, the greater the total technical debt.'
#     ]
#
#     query_emb = model.encode(query)
#     docs_emb = model.encode(docs)
#     scores = util.dot_score(query_emb, docs_emb)[0]
#     doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
#
#     print("Query:", query)
#     for doc, score in doc_scores:
#         # print(doc, score)
#         print(f"{score:0.02f}\t{doc}")
#
#
# print("Original Model")
# show_examples(org_model)
#
# doc_dir = "/app/killer-bots/killer_bots/bots/code_guru/database"
# paragraphs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)
# print(f"Number of paragraphs: {len(paragraphs)}")
# docs = preprocess_docs(paragraphs)
# print(f"Number of docs: {len(docs)}")
# corpus = [clean_wiki_text(doc.content) for doc in docs] + [clean_wiki_text(doc.content) for doc in paragraphs]
# corpus = set(corpus)
# print("Len Corpus:", len(corpus))
#
# try:
#     os.remove("faiss_document_store_gpl.db")
# except:
#     pass
#
# document_store = FAISSDocumentStore(
#     sql_url="sqlite:///faiss_document_store_gpl.db",
#     faiss_index_factory_str="Flat",
#     # similarity="cosine"
# )
# document_store.write_documents([{"content": t} for t in corpus])
#
# retriever = EmbeddingRetriever(
#     document_store=document_store,
#     embedding_model=f"sentence-transformers/{model_name}",
#     model_format="sentence_transformers",
#     # max_seq_len=max_seq_length,
#     progress_bar=False,
# )
# document_store.update_embeddings(retriever)
#
# use_question_generator = True
#
# if use_question_generator:
#     questions_producer = QuestionGenerator(
#         # model_name_or_path="doc2query/msmarco-t5-base-v1",
#         # max_length=64,
#         # split_length=128,
#         batch_size=32,
#         num_queries_per_doc=10,
#     )
#
# else:
#     questions_producer = query_doc_pairs
#
# # We can use either QuestionGenerator or already generated questions in PseudoLabelGenerator
# psg = PseudoLabelGenerator(questions_producer, retriever, max_questions_per_document=10, batch_size=32, top_k=10)
# output, pipe_id = psg.run(documents=document_store.get_all_documents())
#
# retriever.train(output["gpl_labels"])
#
# print("Original Model")
# show_examples(org_model)
#
# print("\n\nAdapted Model")
# show_examples(retriever.embedding_encoder.embedding_model)
#
# retriever.save("adapted_retriever")

model = SentenceTransformer("./adapted_retriever")
model.save_to_hub("AlekseyKorshuk/retriever-coding-guru-adapted", exist_ok=True)