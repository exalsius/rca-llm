import logging

from lib.metrics.metrics import ChunkMetadata
from lib.provider.base_provider import BaseProvider

logger = logging.getLogger("llm-benchmark")


class TritonInferProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "ensemble"

    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        assert not self.parsed_options.stream, "Stream is not supported"
        return f"/v2/models/{self.model}/infer"

    def format_payload(self, prompt, max_tokens):
        # matching latest TRT-LLM example, your model configuration might be different
        data = {
            "inputs": [
                {
                    "name": "text_input",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [[prompt]],
                },
                {
                    "name": "max_tokens",
                    "datatype": "UINT32",
                    "shape": [1, 1],
                    "data": [[max_tokens]],
                },
                {
                    "name": "bad_words",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [[""]],
                },
                {
                    "name": "stop_words",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [[""]],
                },
                {
                    "name": "temperature",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [[self.parsed_options.temperature]],
                },
            ]
        }
        assert self.parsed_options.logprobs is None, "logprobs are not supported"
        return data

    def parse_output_json(self, data, prompt):
        for output in data["outputs"]:
            if output["name"] == "text_output":
                assert output["datatype"] == "BYTES"
                assert output["shape"] == [1]
                text = output["data"][0]
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
        raise ValueError("text_output not found in the response")
