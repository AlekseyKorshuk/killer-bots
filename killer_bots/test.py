import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from killer_bots.bots.code_guru import prompts
from killer_bots.bots.code_guru.bot import CodeGuruBot, CodeGuruBotWithContext, CodeGuruBotLFQA, CodeGuruBotWithDialogue

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = 'facebook/opt-30b'

# MODEL = 'facebook/opt-125m'


# MODEL = 'EleutherAI/gpt-j-6B'


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


params = {
    "top_p": 1,
    "top_k": 20,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    # "eos_token_id": 50118,  # 50118
    "device": device,
    "do_sample": True,
    "max_new_tokens": 256,
}


def load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(MODEL)


if __name__ == "__main__":
    model = load_huggingface_model(MODEL)
    tokenizer = load_tokenizer(MODEL)

    # num_added_toks = tokenizer.add_tokens(['[EOT]'], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))
    # print(tokenizer('[EOT]'))
    # eot_id = tokenizer('[EOT]')['input_ids'][-1]
    # params['eos_token_id'] = 2

    bot = CodeGuruBotWithDialogue(
        model=model,
        tokenizer=tokenizer,
        description={'model': MODEL, 'reward_model': None},
        prompt=prompts.PROMPT,
        max_history_size=3,
        **params,
    )

    response = bot.respond("Can you list all SOLID principles separating them with a new line?")
    print(response)
