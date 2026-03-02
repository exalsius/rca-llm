from lib.provider.openai_provider import OpenAIProvider


class VllmProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens):
        data = super().format_payload(prompt, max_tokens)
        data["ignore_eos"] = True
        return data
