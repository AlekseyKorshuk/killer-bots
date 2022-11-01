import tqdm
from haystack import Document
from haystack.utils import convert_files_to_docs
from sentence_transformers import SentenceTransformer, util
from setfit import SetFitModel
import re

from killer_bots.search_engine.utils import change_extentions_to_txt


def clean_wiki_text(text: str) -> str:
    # get rid of multiple new lines
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    # get rid of multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")

    # add paragraphs (identified by wiki section title which is always in format "==Some Title==")
    text = text.replace("\n==", "\n\n\n==")

    # remove empty paragrahps
    text = re.sub(r"(==.*==\n\n\n)", "", text)

    # remove multiple dashes
    text = re.sub(r"#+ +", "", text)
    text = re.sub(r" +#+", "", text)

    text = re.sub(r"\n +\n", "\n", text)

    return text.strip()


def get_docs_text(docs):
    return '\n'.join([clean_wiki_text(doc.content) for doc in docs])


def join_docs(docs):
    if set([doc.meta["name"] for doc in docs]) == 1:
        print(docs)
        raise Exception("All docs must have the same name")
    return Document(content=get_docs_text(docs), meta=docs[0].meta)


class PreprocessDocs:
    def __init__(self):
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.is_title_model = SetFitModel.from_pretrained("AlekseyKorshuk/is-title-setfit", device="cuda")
        self.small_threshold = 0.1
        self.threshold = 0.35
        self.next_threshold = 0.4

    def get_score(self, text1, text2):
        text1 = clean_wiki_text(text1)
        text2 = clean_wiki_text(text2)
        embeddings = self.similarity_model.encode([text1, text2], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        if "technical debt" in text1 or "technical debt" in text2:
            print("#" * 100)
            print(text1)
            print(text2)
            print(cosine_scores[0][1])
            input()

        return cosine_scores[0][1]

    def is_title(self, text):
        return self.is_title_model([clean_wiki_text(text)])[0] == 1

    def split_last_titles(self, docs):
        last_docs = []
        doc = docs[-1]
        while self.is_title(doc.content):
            last_docs.append(doc)
            docs.pop()
            doc = docs[-1]
        last_docs.reverse()
        return docs, last_docs

    def __call__(self, docs):
        prepared_docs = []
        current_docs = [docs[0]]
        for i, doc in tqdm.tqdm(enumerate(docs[1:]), total=len(docs[1:]), desc="Preprocessing docs"):
            if doc.meta['name'] != current_docs[-1].meta['name']:
                prepared_docs.append(join_docs(current_docs))
                current_docs = [doc]
                continue

            score = self.get_score(get_docs_text(current_docs), doc.content)
            add_flag = False
            if score > self.threshold and current_docs[-1].meta['name'] == doc.meta['name']:
                add_flag = True
            elif self.is_title(get_docs_text(current_docs)) and score > self.small_threshold:
                add_flag = True
            else:
                next_score = self.get_score(
                    get_docs_text(current_docs) + '\n' + doc.content,
                    docs[i + 2].content
                )
                if next_score > self.next_threshold and current_docs[-1].meta['name'] == docs[i + 2].meta['name']:
                    add_flag = True
            if add_flag:
                current_docs.append(doc)
                if i == len(docs) - 2:
                    prepared_docs.append(join_docs(current_docs))
            else:
                current_docs, last_docs = self.split_last_titles(current_docs)
                prepared_docs.append(join_docs(current_docs))
                current_docs = last_docs + [doc]
        return prepared_docs


def preprocess_docs(docs):
    return PreprocessDocs()(docs)

if __name__ == "__main__":
    doc_dir = './killer_bots/bots/code_guru/database/'
    change_extentions_to_txt(doc_dir)
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)
    preprocessor = PreprocessDocs()
    prepared_docs = preprocessor(docs)

    for doc in prepared_docs:
        print(doc)
        print("#" * 100)