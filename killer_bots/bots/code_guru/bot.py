from killer_bots.bots.base import Bot
from killer_bots.bots.code_guru import prompts


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

    def _format_model_inputs(self, text):
        lines = [prompts.START_PROMPT] + self.chat_history
        lines += ["Context: " + prompts.SOLID_CONTEXT]
        lines += [f"{self.bot_name}[answers based on context]:"]
        lines = "\n".join(lines)
        print("PROMPT:")
        print(lines)
        print("END PROMPT")
        return lines

