import abc
import copy


class BaseProvider(abc.ABC):
    DEFAULT_MODEL_NAME = None

    def __init__(self, model, parsed_options):
        self.model = model
        self.parsed_options = copy.deepcopy(parsed_options)
        if self.parsed_options.logprobs is not None:
            if self.parsed_options.chat:
                self.parsed_options.logprobs = bool(
                    self.parsed_options.logprobs == "true"
                )
            else:
                self.parsed_options.logprobs = int(self.parsed_options.logprobs)

    @abc.abstractmethod
    def get_url(self):
        ...

    @abc.abstractmethod
    def format_payload(self, prompt, max_tokens):
        ...

    @abc.abstractmethod
    def parse_output_json(self, json, prompt):
        ...
