import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from killer_bots.bots.code_guru import prompts
from killer_bots.evaluation.utils import run_score
from killer_bots.bots.code_guru.bot import CodeGuruBot, CodeGuruBotWithContext, CodeGuruBotLFQA, CodeGuruBotWithDialogue

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = 'facebook/opt-30b'
# MODEL = 'facebook/opt-125m'
# MODEL = 'EleutherAI/gpt-j-6B'

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

    run_score(
        [
            CodeGuruBotWithDialogue(
                model=model,
                tokenizer=tokenizer,
                description={'model': MODEL, 'reward_model': None},
                prompt=prompts.PROMPT,
                max_history_size=3,
                **params,
            ),
        ],
        do_save=True
    )
