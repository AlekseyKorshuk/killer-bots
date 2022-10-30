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


def nest_sentences(document):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 1024:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = []
            length = 0

    if sent:
        nested.append(sent)

    return nested


nested = nest_sentences(DOCUMENT)

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

BART_PATH = 'facebook/bart-large-cnn'

bart_model = BartForConditionalGeneration.from_pretrained(BART_PATH, output_past=True)
bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH, output_past=True)


def generate_summary(nested_sentences):
    device = 'cuda'
    summaries = []
    for nested in nested_sentences:
        input_tokenized = bart_tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
        input_tokenized = input_tokenized.to(device)
        summary_ids = bart_model.to('cuda').generate(input_tokenized,
                                                     length_penalty=3.0,
                                                     min_length=30,
                                                     max_length=100)
        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]
        summaries.append(output)
    summaries = [sentence for sublist in summaries for sentence in sublist]
    return summaries


sm = Summarizer(model='distilbert-base-uncased')


def generate_summary2(nested_sentences):
    device = 'cuda'
    summaries = []
    for nested in nested_sentences:
        k = sm.calculate_optimal_k(' '.join(nested), k_max=10)
        output = sm(body=' '.join(nested), num_sentences=k)

        # input_tokenized = bart_tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
        # input_tokenized = input_tokenized.to(device)
        # summary_ids = bart_model.to('cuda').generate(input_tokenized,
        #                                   length_penalty=3.0,
        #                                   min_length=30,
        #                                   max_length=100)
        # output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        summaries.append(output)
    return summaries


summ = generate_summary(nested)

print("\n".join(summ))

print(len(DOCUMENT))

summ = generate_summary2(nested)
print("\n".join(summ))

res = sm.calculate_optimal_k(DOCUMENT, k_max=10)
print(res)
