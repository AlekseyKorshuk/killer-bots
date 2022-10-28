import os

import torch
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification, pipeline,
    AutoModelForSeq2SeqLM
)

from killer_bots.bots.code_guru import prompts
from killer_bots.bots.code_guru.bot import CodeGuruBot, CodeGuruBotWithContext, CodeGuruBotLFQA, CodeGuruBotWithDialogue
import pandas as pd
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = 'facebook/opt-30b'
# MODEL = 'facebook/opt-125m'
# MODEL = 'EleutherAI/gpt-j-6B'

REWARD_MODEL = "ChaiML/roberta-base-dalio-reg-v1"
# REWARD_MODEL = '/models/dalio_reward_models/checkpoint-2700'

REWARD_TOKENIZER = "ChaiML/roberta-base-dalio-reg-v1"


# REWARD_TOKENIZER = 'EleutherAI/gpt-neo-2.7B'


def load_huggingface_model(model_id, model_class=AutoModelForCausalLM):
    filename = model_id.replace('/', '_').replace('-', '_') + '.pt'
    cache_path = os.path.join('/tmp', filename)

    start = time.time()
    if os.path.exists(cache_path):
        print('loading model from cache')
        model = torch.load(cache_path)
    else:
        print('loading model from scratch')
        model = model_class.from_pretrained(model_id)
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
    "eos_token_id": 50118,  # 50118
    "device": device,
    "do_sample": True,
    "max_new_tokens": 256,
}


def load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(MODEL)


TEST_QUESTIONS = [
    "What is SOLID?",
    "What is Single Responsibility Principle?",
    "What is Open-Closed Principle?",
    "What is Liskov Substitution Principle?",
    "What is Interface Segregation Principle?",
    "What is Dependency Inversion Principle?",
    "Why should I use SOLID?",
    "Who created SOLID?",
]


def get_evaluation_pipeline():
    pipe = pipeline("text2text-generation", model="google/flan-t5-xl", device=0,
                    model_kwargs={"torch_dtype": torch.bfloat16})
    return pipe


def hypothesis_call(pipe, context, response):
    prompt = "Premise:\n" \
             "{}\n" \
             "Hypothesis:\n" \
             "{}\n" \
             "Does the premise entail the hypothesis (yes/no)?"
    prompt = prompt.format(context, response)
    response = str(pipe(prompt, max_length=1024)[0]["generated_text"]).lower().strip()
    if "yes" in response:
        return 2
    elif "no" in response:
        return 0
    return 1


if __name__ == "__main__":
    model = load_huggingface_model(MODEL)
    tokenizer = load_tokenizer(MODEL)

    bot = CodeGuruBotWithDialogue(
        model=model,
        tokenizer=tokenizer,
        description={'model': MODEL, 'reward_model': None},
        prompt=prompts.PROMPT,
        max_history_size=3,
        **params,
    )

    TEST_QUESTIONS = TEST_QUESTIONS * 5
    pipe = get_evaluation_pipeline()
    stats = []
    for question in tqdm.tqdm(TEST_QUESTIONS):
        response = bot.respond(question)
        context = bot.previous_context[-1]
        # bot.previous_context = []
        result = hypothesis_call(pipe, context, response)
        print("Question:", question)
        print("Context:", context)
        print("Response:", response)
        print("Result:", result)
        stats.append(int(result))
    stats = np.array(stats)
    df = pd.DataFrame(stats, index=TEST_QUESTIONS, columns=['result'])
    print(df.describe())
