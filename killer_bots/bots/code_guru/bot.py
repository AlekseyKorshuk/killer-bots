from killer_bots.bots.base import Bot
from killer_bots.bots.code_guru import prompts
from killer_bots.search_engine.search_summarization import SearchSummarization


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
        self.pipeline = SearchSummarization("/app/killer-bots/killer_bots/bots/code_guru/database")

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
