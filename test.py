import re
import nltk
from summarizer import Summarizer

nltk.download('punkt')
nltk.download('stopwords')

file_path = './killer_bots/bots/code_guru/database/handwritten.txt'
with open(file_path, 'r') as f:
    data = f.read()
DOCUMENT = data

DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
DOCUMENT = DOCUMENT.strip()

sm = Summarizer(model='distilbert-base-uncased')
result = sm(body=DOCUMENT, ratio=0.15)
print(result)
