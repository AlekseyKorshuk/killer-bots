from haystack.utils import convert_files_to_docs
from rakun2 import RakunKeyphraseDetector

hyperparameters = {
    "num_keywords": 10,
    # "merge_threshold": 1.1,
    # "alpha": 0.3,
    # "token_prune_len": 3
}

keyword_detector = RakunKeyphraseDetector(hyperparameters)

doc_dir = "/app/killer-bots/killer_bots/bots/code_guru/database"
paragraphs = convert_files_to_docs(dir_path=doc_dir, clean_func=None, split_paragraphs=True)

for paragraph in paragraphs:
    keywords = keyword_detector.find_keywords(paragraph.content, input_type="string")
    print(paragraph.content)
    print(keywords)
    print("#" * 80)
