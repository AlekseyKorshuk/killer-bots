import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from killer_bots.bots.base import Bot
from killer_bots.bots.therapist import prompts
from killer_bots.search_engine.custom_pipeline import Pipeline
from killer_bots.search_engine.search_query import SearchQueryGenerator
from killer_bots.search_engine.search_summarization import SearchSummarization
from killer_bots.search_engine.lfqa import LFQA
from killer_bots.search_engine.web_parser import GoogleSearchEngine, GoogleSearchEngine2
from killer_bots.bots.therapist.utils import get_search_pipeline

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


class TherapistBotGoogleSearch(Bot):
    def __init__(self, model, tokenizer, description, **params):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            description=description,
            bot_name="Therapist",
            first_message="I am happy to help with any problems. What situation are you facing?",
            **params,
        )
        self.pipeline = GoogleSearchEngine()
        self.stop_words = ["User:", "Therapist:", "Context:"]
        stopping_criterias = []
        for word in self.stop_words:
            input_ids = self.tokenizer(word, add_special_tokens=False).input_ids
            stopping_criterias.append(
                StoppingCriteriaSub(stops=input_ids)
            )
        self.stopping_criteria = StoppingCriteriaList(stopping_criterias)
        self.top_k = 1
        self.search_history = ["none"]
        self.search_query_generator = SearchQueryGenerator()
        self.chat_search_pipeline = get_search_pipeline()

    def _generate_sample_chats(self, query, num_samples=2, num_turns=4):
        total_num_docs = num_samples * num_turns
        documents = self.chat_search_pipeline.run(
            query=query,
            params={
                "Retriever": {
                    "top_k": total_num_docs
                }
            }
        )["documents"]
        documents.reverse()
        chats = []
        for i in range(num_samples):
            chat = []
            for j in range(num_turns):
                chat.append(documents[i * num_turns + j].content)
            chats.append("\n".join(chat))
        return chats

    def _format_model_inputs(self, text):
        self.search_history = self.search_history[:len(self.chat_history) // 2]
        print("Generating search query...")
        # search_query = self.search_query_generator(self.chat_history, self.search_history, self.model, self.tokenizer)
        search_query = "none"
        print("Search query generated:", search_query)
        self.search_history.append(search_query)
        context = "empty"
        if search_query != "none":
            print("Parsing search query...")
            search_query = f"therapist, {search_query}"
            # context = self.pipeline(search_query, top_k=self.top_k)
            # context = "\n".join(context)
            context = "empty"

        chats = self._generate_sample_chats("\n".join(self._get_cropped_history()), num_samples=2, num_turns=4)
        lines = [prompts.EXAMPLE_TITLE, ""]
        for chat in chats:
            lines.append(chat)
            lines.append("")
        # lines += [prompts.START_TEMPLATE.format(context)]
        lines += self._get_cropped_history()
        lines += [f"{self.bot_name}:"]
        lines = "\n".join(lines)
        print("PROMPT:")
        print(lines)
        print("END PROMPT")
        print("SEARCH QUERY:", search_query)
        return lines
