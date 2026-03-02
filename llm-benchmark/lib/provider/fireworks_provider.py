from lib.provider.openai_provider import OpenAIProvider


class FireworksProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens):
        data = super().format_payload(prompt, max_tokens)
        data["min_tokens"] = max_tokens
        data["prompt_cache_max_len"] = 0
        return data
