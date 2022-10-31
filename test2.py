import tqdm
from haystack.utils import convert_files_to_docs
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util

# Sentences we want sentence embeddings for
from killer_bots.search_engine.custom_pipeline import clean_wiki_text
from killer_bots.search_engine.utils import change_extentions_to_txt

doc_dir = './killer_bots/bots/code_guru/database/'

change_extentions_to_txt(doc_dir)
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)
sentences = [doc.content for doc in docs]

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_score(text1, text2):
    # Compute embedding for both lists
    text1 = clean_wiki_text(text1)
    text2 = clean_wiki_text(text2)
    # Compute embeddings
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)

    return cosine_scores[0][1]


threshold = 0.35
final_docs = []
current_doc = docs[0]
for i, doc in tqdm.tqdm(enumerate(docs[1:]), total=len(docs[1:])):
    score = get_score(current_doc.content, doc.content)

    add_flag = False
    if score > threshold and current_doc.meta['name'] == doc.meta['name']:
        add_flag = True
    # else:
    #     next_score = get_score(
    #         current_doc.content + '\n' + doc.content,
    #         docs[i + 1].content
    #     )
    #     if next_score > threshold and current_doc.meta['name'] == docs[i + 1].meta['name']:
    #         add_flag = True
    if add_flag:
        current_doc.content += "\n" + doc.content
        if i == len(docs) - 2:
            final_docs.append(current_doc)
    else:
        final_docs.append(current_doc)
        current_doc = doc

for doc in final_docs:
    print(doc.content)
    print("#" * 100)

# pairs = []
# for i in range(len(sentences) - 1):
#     pairs.append((i, i + 1))
#
# embeddings = model.encode(sentences, convert_to_tensor=True)
# cosine_scores = util.cos_sim(embeddings, embeddings)
# print(cosine_scores)
#
# for i, sentence in enumerate(sentences):
#     print("Sentence {}:".format(i))
#     print(sentence)
#     print("")
#
# for pair in pairs:
#     print("Sentence {}: {}".format(pair[0], sentences[pair[0]]))
#     print("Sentence {}: {}".format(pair[1], sentences[pair[1]]))
#     print("Similarity between sentence {} and sentence {} is: {:.4f}".format(pair[0], pair[1],
#                                                                              cosine_scores[pair[0], pair[1]]))
#     print("")
