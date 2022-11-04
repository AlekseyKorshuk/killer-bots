import torch
import logging


class Bot(object):
    def __init__(self, model, tokenizer, description, **params):
        self.model = model
        self.tokenizer = tokenizer
        self.description = description

        self.tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.eos_token_id = params.get("eos_token_id", 198)

        self.bot_name = params.get("bot_name", "Bot")
        self.user_name = params.get("user_name", "User")
        self.first_message = params.get("first_message", "Hi.")
        self.prompt = params.get("prompt", "This is a conversation.")
        self.max_history_size = params.get("max_history_size", -1)

        self.temperature = params.get("temperature", 0.72)
        self.top_p = params.get("top_p", 0.725)
        self.top_k = params.get("top_k", 0)
        self.do_sample = params.get("do_sample", True)
        self.max_new_tokens = params.get("max_new_tokens", 64)
        self.repetition_penalty = params.get("repetition_penalty", 1.13125)
        self.stopping_ids = None
        self.stopping_criteria = None
        self.stop_words = []

        self.device = params.get("device", 0)
        self.reset_chat_history()

    def _get_cropped_history(self):
        if self.max_history_size > 0 and len(self.chat_history) > self.max_history_size * 2:
            return self.chat_history[:1] + self.chat_history[-self.max_history_size * 2 - 1:]
        return self.chat_history

    def reset_chat_history(self):
        self.chat_history = []
        self._add_bot_message(self.first_message)

    def _get_request_params(self):
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            # "eos_token_id": self.eos_token_id,
            # "pad_token_id": self.eos_token_id,
        }

    def _format_model_inputs(self, text):
        lines = [self.prompt] + self.chat_history + [f"{self.bot_name}:"]
        return "\n".join(lines)

    def _tokenize(self, inputs, padding=False):
        # print('***')
        # print(inputs[0])
        # print('***')

        self._print_num_tokens(inputs, padding)

        max_length = 1900
        tokenized_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=padding,
            max_length=max_length,
        ).to(self.device)

        if len(tokenized_inputs.input_ids[0]) > max_length:
            print()
            print('input is being truncated!!')
            print()
        return tokenized_inputs

    def _print_num_tokens(self, inputs, padding):
        tokenized_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=False,
            padding=padding,
        )
        print('***')
        print('num tokens', len(tokenized_inputs.input_ids[0]))
        print('***')

    @torch.inference_mode()
    def _generate(self, encoded, request_params):
        print(request_params)
        return self.model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            stopping_criteria=self.stopping_criteria,
            **request_params,
        )

    def _decode(self, outputs):
        decoded = self.tokenizer.decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print("DECODED: ", decoded)
        decoded = decoded.rstrip()
        for stop_word in self.stop_words:
            decoded = decoded.replace(stop_word, "")
        return decoded

    def _add_user_message(self, text):
        self.chat_history.append(f"{self.user_name}: {text.strip()}")

    def _add_bot_message(self, response):
        self.chat_history.append(f"{self.bot_name}: {response.strip()}")

    def _truncate_incomplete_sentence(self, response):
        for i in reversed(range(len(response))):
            if response[i] in [".", "!", "?", "*", "\n"]:
                break
        return response if not response or i < 1 else response[:i + 1]

    def get_context(self):
        return ""

    def respond(self, text):
        self._add_user_message(text)

        request_params = self._get_request_params()
        inputs = [self._format_model_inputs(text)]

        encoded = self._tokenize(inputs)
        outputs = self._generate(encoded, request_params)

        print("Output length:", len(outputs[0][len(encoded.input_ids[0]):]))
        decoded = self._decode(outputs[0][len(encoded.input_ids[0]):])
        response = self._truncate_incomplete_sentence(decoded.rstrip())

        self._add_bot_message(response)
        return response
