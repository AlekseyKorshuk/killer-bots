import datetime
import os
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

USERNAME = os.environ['NAME']

OUTPUT_DIR = '/models/dalio_bot_people_scores'

CATEGORY = {
    "0": "useless",
    "1": "acceptable",
    "2": "helpful",
    "3": "very helpful",
    "8": "quit conversation",
    "9": "write your own"
}

NUM_MESSAGES_PER_CONVERSATION = 300


def get_save_path():
    time_format = '%Y%m%d_%H%M%S'
    timestamp = str(datetime.datetime.utcnow().strftime(time_format))
    assert USERNAME is not None, 'invalid username: {} (see ab_testing.py)'.format(USERNAME)
    filename = '{}_{}_{}.json'.format(USERNAME, timestamp, NUM_MESSAGES_PER_CONVERSATION)
    path = os.path.join(OUTPUT_DIR, filename)
    print('saving results to {}'.format(path))
    return path


def run_score(bots, do_save=False):
    path = get_save_path() if do_save else None

    scores = [0 for bot in bots]
    while True:
        indices = np.random.permutation(len(bots))
        for i in indices:
            score = run_conversation(bots[i], save_path=path)
            scores[i] += score
            if path is not None:
                history = "\n".join(bots[i].chat_history)
                whole_score = get_whole_score()
                write_whole_conversation_score(history, whole_score, bots[i].description, path)
        print(scores)


def run_conversation(bot, save_path=None):
    print("[START]")
    bot.reset_chat_history()

    responses = []
    conversation_score = 0
    for i in range(NUM_MESSAGES_PER_CONVERSATION):
        history = "\n".join(bot.chat_history)
        print(history)
        text = input("ENTER MESSAGE: ")

        score = "0"
        while score == "0":
            response = bot.respond(text)

            history = "\n".join(bot.chat_history)

            print(f"[MESSAGE {i + 1}]")
            print(history)

            score = get_score()

            if score == "9":
                response = input("Write your own response: ").strip()
                # remove bot response
                bot.chat_history.pop()
                bot._add_bot_message(response)

                history = "\n".join(bot.chat_history)
                print(f"[MESSAGE {i + 1}]")
                print(history)
                score = get_score()
                assert score not in ["9"]

            conversation_score += float(score)

            if save_path is not None:
                write_conversation_score(history, score, bot.description, save_path)
            if score == "8":
                return conversation_score

            if score == "0":
                print('retrying last bot message...')
                # remove last bot message
                bot.chat_history.pop()
                # remove last user message
                bot.chat_history.pop()

    print("[END]")
    return conversation_score


def get_score():
    string = '; '.join(['{}={}'.format(k, v) for k, v in CATEGORY.items()])
    score = input('SCORE RESPONSE: {} '.format(string))
    score = score.strip()

    if score not in CATEGORY.keys():
        print("Unknown category: '{}'".format(score))
        score = get_score()

    return score


def get_whole_score():
    string = '; '.join(['{}={}'.format(k, v) for k, v in CATEGORY.items()])
    score = input('SCORE THE WHOLE CONVERSATION: {} '.format(string))
    score = score.strip()

    if score not in CATEGORY.keys():
        print("Unknown category: '{}'".format(score))
        score = get_score()

    return score


def write_responses(text, responses, choice, path="./responses.jsonl"):
    with open(path, "a") as f:
        f.write(
            json.dumps({"text": text, "responses": responses, "choice": choice}) + "\n"
        )


def write_score(text, response, score, path="./scores.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps({"text": text, "response": response, "score": score}) + "\n")


def write_conversation_score(text, score, description, path="./conversation_scores.jsonl"):
    with open(path, "a") as f:
        data = {"text": text, "label": CATEGORY[score], "score": score, 'model': description, 'type': 'response'}
        f.write(json.dumps(data) + "\n")


def write_whole_conversation_score(text, score, description, path="./conversation_scores.jsonl"):
    with open(path, "a") as f:
        data = {"text": text, "label": CATEGORY[score], "score": score, 'model': description,
                'type': 'whole_conversation'}
        f.write(json.dumps(data) + "\n")
