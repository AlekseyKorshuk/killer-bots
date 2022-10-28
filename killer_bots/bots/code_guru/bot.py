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
        StoppingCriteria.__init__(self)
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        input_ids = input_ids.cpu()[0][-len(self.stops):]
        for a, b in zip(input_ids, self.stops):
            if a != b:
                return False
        return True


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
        stop_words = ["User:", "Guru:", "Context:"]
        stopping_criterias = []
        for word in stop_words:
            input_ids = self.tokenizer(word).input_ids[1:]
            stopping_criterias.append(
                StoppingCriteriaSub(stops=input_ids)
            )
        self.stopping_criteria = StoppingCriteriaList(stopping_criterias)
        self.previous_context_size = 2
        self.previous_context = []

    def get_context(self):
        self.previous_contex = self.previous_context[-self.previous_context_size:]
        context = self.previous_context
        context.reverse()
        context = "\n".join(set(context))
        return context

    def _format_model_inputs(self, text):
        current_context = self.pipeline(text)
        self.previous_context.append(current_context)
        context = self.get_context()
        lines = [prompts.START_TEMPLATE.format(context)]
        lines += self._get_cropped_history()
        lines += [f"{self.bot_name}:"]
        lines = "\n".join(lines)
        print("PROMPT:")
        print(lines)
        print("END PROMPT")
        # print("CHAT HISTORY:")
        # print(self.chat_history)
        # print("END CHAT HISTORY")
        return lines
