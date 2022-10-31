import tqdm
from haystack.utils import convert_files_to_docs
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import datasets
# Sentences we want sentence embeddings for
from killer_bots.search_engine.custom_pipeline import clean_wiki_text
from killer_bots.search_engine.utils import change_extentions_to_txt

doc_dir = './killer_bots/bots/code_guru/database/'

change_extentions_to_txt(doc_dir)
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)
sentences = [doc.content.strip() for doc in docs]

# data = [
#     {
#         'text': clean_wiki_text(sentence),
#         'label': 1 if str(sentence).startswith("#") else 0
#     } for sentence in sentences
# ]

from sklearn.model_selection import train_test_split



data = {
    "text": [clean_wiki_text(sentence) for sentence in sentences],
    "label": [1 if str(sentence).startswith("#") else 0 for sentence in sentences]
}

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2)

print(y_test)
print('% of positive samples in train set:', sum(y_train) / len(X_train))
print('% of positive samples in test set:', sum(y_test) / len(X_test))

# dataset = datasets.Dataset.from_dict(data)

dataset = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_dict(
            {
                "sentence": X_train,
                "label": y_train
            }
        ),
        "validation": datasets.Dataset.from_dict(
            {
                "sentence": X_test,
                "label": y_test
            }
        )
    }
)

dataset.push_to_hub("AlekseyKorshuk/is_title")