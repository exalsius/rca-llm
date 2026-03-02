import logging

from lib.metrics.metrics import ChunkMetadata
from lib.provider.base_provider import BaseProvider

logger = logging.getLogger("llm-benchmark")


class TritonGenerateProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "ensemble"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        stream_suffix = "_stream" if self.parsed_options.stream else ""
        return f"/v2/models/{self.model}/generate{stream_suffix}"

    def format_payload(self, prompt, max_tokens):
        data = {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
            # for whatever reason these has to be provided
            "bad_words": "",
            "stop_words": "",
        }
        assert self.parsed_options.logprobs is None, "logprobs are not supported"
        return data

    def parse_output_json(self, data, prompt):
        text = data["text_output"]
        if not self.parsed_options.stream:
            # Triton returns the original prompt in the output, cut it off
            text = text.removeprefix("<s> ")
            if text.startswith(prompt):
                # HF tokenizers get confused by the leading space
                text = text[len(prompt) :].removeprefix(" ")
            else:
                logger.warning("WARNING: prompt not found in the output")
        return ChunkMetadata(
            text=text,
            logprob_tokens=None,
            usage_tokens=None,
            prompt_usage_tokens=None,
        )
