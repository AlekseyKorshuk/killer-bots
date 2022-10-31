import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = 'facebook/opt-30b'
# MODEL = 'facebook/opt-125m'
MODEL = 'EleutherAI/gpt-j-6B'

REWARD_MODEL = "ChaiML/roberta-base-dalio-reg-v1"
# REWARD_MODEL = '/models/dalio_reward_models/checkpoint-2700'

REWARD_TOKENIZER = "ChaiML/roberta-base-dalio-reg-v1"


# REWARD_TOKENIZER = 'EleutherAI/gpt-neo-2.7B'


def load_huggingface_model(model_id):
    filename = model_id.replace('/', '_').replace('-', '_') + '.pt'
    cache_path = os.path.join('/tmp', filename)

    start = time.time()
    if os.path.exists(cache_path):
        print('loading model from cache')
        model = torch.load(cache_path)
    else:
        print('loading model from scratch')
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.half().eval().to(device)
        torch.save(model, cache_path)
    print('duration', time.time() - start)
    return model


# reward_model = (
#    AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL)
#    .eval()
#    .to(device)
# )
# reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_TOKENIZER)
# reward_tokenizer.pad_token = reward_tokenizer.eos_token
# reward_model.config.pad_token_id = reward_tokenizer.pad_token_id


params = {
    "top_p": 0.5703324165138496,
    "top_k": 8,
    "temperature": 0.9758783896898277,
    "repetition_penalty": 1.0302804540995645,
    # "eos_token_id": 50118,  # 50118
    "device": device,
    "do_sample": True,
    "max_new_tokens": 256,
}

params = {
    # "top_p": 1,
    # "top_k": 20,
    # "temperature": 1.0,
    # "repetition_penalty": 1.0,
    "eos_token_id": 198,  # 50118
    "device": device,
    "do_sample": False,
    "max_new_tokens": 256,
}



def load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(MODEL)


prompt = """This is a conversation where Coding Guru is giving advice on being a developer and helping with any questions. Guru uses Search Engine to find needed information.

User: Hi Guru, thanks for talking with me today. I am excited to learn more about how to be a good developer and solve some problems.
Search: none
Bot: I am happy to help with any coding problem. What situation are you facing?
User: I think I am bad developer because I can not write good code. What should I do?
Search: clean code, how to write clean code, how to be a good developer, advice
Bot: This is very good question. The problem of not writing good code is common. A good start would be reading about Refactoring, OOP, SOLID, Design Patterns. I would recommend reading Clean Code by Robert C. Martin. It is a great book that will help you write better code. 
User: What is Refactoring?
Search: refactoring, what is refactoring, why we need to refactor code
Bot: Refactoring is a process of improving the quality of the code. It’s a way to make the code more readable, maintainable, and extensible.
Search: refactoring, clean code, tools/tips to write clean code
Bot: You can use refactoring to clean up the code. Refactoring is a process of improving the quality of the code.
User: How I can refactor my code?
Search: refactoring, how to refactor code, tools to refactor code
Bot: Refactoring should be done as a series of small changes, each of which makes the existing code slightly better while still leaving the program in working order.
User: Thanks Guru, I will try to refactor my code.
Search: none
Bot: You are welcome. I am happy to help.

This is a conversation where Coding Guru is giving advice on being a developer and helping with any questions. Guru uses Search Engine to find needed information.

User: Hi Guru, thanks for talking with me today. I am excited to learn more about how to be a good developer and solve some problems.
Search: none
Bot: I am happy to help with any coding problem. What situation are you facing?
User: """

if __name__ == "__main__":
    model = load_huggingface_model(MODEL)
    tokenizer = load_tokenizer(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    while True:
        query = input("User: ")
        if query == "exit":
            break
        inputs = tokenizer(prompt + query + "\nSearch:", return_tensors="pt", padding=False).to(device)
        output_ids = model.generate(
            **inputs, **params
        )
        output_text = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"Search:{output_text}")

