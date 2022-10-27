import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from killer_bots.bots.base import Bot
from killer_bots.bots.code_guru import prompts
from killer_bots.search_engine.search_summarization import SearchSummarization
from killer_bots.search_engine.lfqa import LFQA


class CodeGuruBot(Bot):
    def __init__(self, model, tokenizer, description, **params):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            description=description,
            bot_name="CodeGuru",
            first_message="I am happy to help with any coding problem. What situation are you facing?",
            **params,
        )


class CodeGuruBotWithContext(Bot):
    def __init__(self, model, tokenizer, description, **params):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            description=description,
            bot_name="Guru",
            first_message="I am happy to help with any coding problem. What situation are you facing?",
            **params,
        )
        self.pipeline = LFQA("/app/killer-bots/killer_bots/bots/code_guru/database")

    def _format_model_inputs(self, text):
        lines = [self.prompt] + self.chat_history
        context = self.pipeline(text)
        lines += ["Context: " + context]
        lines += [f"{self.bot_name}:"]
        lines = "\n".join(lines)
        print("PROMPT:")
        print(lines)
        print("END PROMPT")
        return lines


class CodeGuruBotLFQA(Bot):
    def __init__(self, model, tokenizer, description, **params):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            description=description,
            bot_name="Guru",
            first_message="I am happy to help with any coding problem. What situation are you facing?",
            **params,
        )
        self.pipeline = LFQA("/app/killer-bots/killer_bots/bots/code_guru/database")

    def respond(self, text):
        self._add_user_message(text)
        response = self.pipeline(text)
        self._add_bot_message(response)
        return response


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[]):
        StoppingCriteria.__init__(self),

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        self.stops = stops
        for i in range(len(stops)):
            self.stops = self.stops[i]


class CodeGuruBotWithDialogue(Bot):
    def __init__(self, model, tokenizer, description, **params):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            description=description,
            bot_name="Guru",
            first_message="I am happy to help with any coding problem. What situation are you facing?",
            **params,
        )
        self.pipeline = LFQA("/app/killer-bots/killer_bots/bots/code_guru/database")
        input_ids = self.tokenizer("User:").input_ids
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[input_ids])])

    def _format_model_inputs(self, text):
        context = self.pipeline(text)

        lines = [prompts.START_TEMPLATE.format(context)]
        # lines += ["", prompts.START_PROMPT]
        lines += self._get_cropped_history()
        lines += [f"{self.bot_name}:"]
        lines = "\n".join(lines)
        print("PROMPT:")
        print(lines)
        print("END PROMPT")
        print("CHAT HISTORY:")
        print(self.chat_history)
        print("END CHAT HISTORY")
        return lines
