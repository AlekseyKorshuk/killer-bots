from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util

# Sentences we want sentence embeddings for
file_path = './killer_bots/bots/code_guru/database/handwritten.txt'
with open(file_path, 'r') as f:
    data = f.read()

## We split this article into paragraphs and then every paragraph into sentences
sentences = []
for paragraph in data.replace("\r\n", "\n").split("\n\n"):
    if len(paragraph.strip()) > 0:
        for sent in sent_tokenize(paragraph):
            sentences.append(sent)
        # sentences.append(sent_tokenize(paragraph.strip()))

model = SentenceTransformer('all-MiniLM-L6-v2')

pairs = []
for i in range(len(sentences) - 1):
    pairs.append((i, i + 1))

embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)
print(cosine_scores)

for i, sentence in enumerate(sentences):
    print("Sentence {}:".format(i))
    print(sentence)
    print("")


for pair in pairs:
    print("Sentence {}: {}".format(pair[0], sentences[pair[0]]))
    print("Sentence {}: {}".format(pair[1], sentences[pair[1]]))
    print("Similarity between sentence {} and sentence {} is: {:.4f}".format(pair[0], pair[1], cosine_scores[pair[0], pair[1]]))
    print("")

# for pair in pairs:
#     print(pair[0])
#     print(pair[1])
#     embeddings1 = model.encode(pair[0], convert_to_tensor=True)
#     embeddings2 = model.encode(pair[1], convert_to_tensor=True)
#     cosine_scores = util.cos_sim(embeddings1, embeddings2)
#     print(cosine_scores)
# split list into pairs
