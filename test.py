from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_score(text1, text2):
    sentences = [text1, text2]
    # Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)

    return cosine_scores[0][1]


# Single list of sentences
sentences = ['#### During a code review',
             'The code review may be the last chance to tidy up the code before it becomes available to the public.',
             ]

print(get_score(sentences[0], sentences[1]))
