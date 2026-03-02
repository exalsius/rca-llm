from lib.metrics.metrics import ChunkMetadata
from lib.provider.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    def get_url(self):
        if self.parsed_options.chat:
            return "/v1/chat/completions"
        else:
            return "/v1/completions"

    def format_payload(self, prompt, max_tokens):
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
        }
        if self.parsed_options.chat:
            data["messages"] = [{"role": "user", "content": prompt}]
        else:
            data["prompt"] = prompt
        if self.parsed_options.logprobs is not None:
            data["logprobs"] = self.parsed_options.logprobs
        return data

    def parse_output_json(self, data, prompt):
        usage = data.get("usage", None)

        assert len(data["choices"]) == 1, f"Too many choices {len(data['choices'])}"
        choice = data["choices"][0]
        if self.parsed_options.chat:
            if self.parsed_options.stream:
                text = choice["delta"].get("content", "")
            else:
                text = choice["message"]["content"]
        else:
            text = choice["text"]

        if text is None:
            text = ""

        logprobs = choice.get("logprobs", None)
        logprobs_key = "content" if self.parsed_options.chat else "tokens"
        return ChunkMetadata(
            text=text,
            logprob_tokens=len(logprobs[logprobs_key]) if logprobs else None,
            usage_tokens=usage["completion_tokens"] if usage else None,
            prompt_usage_tokens=usage.get("prompt_tokens", None) if usage else None,
        )
