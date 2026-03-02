from lib.provider.openai_provider import OpenAIProvider


class TogetherProvider(OpenAIProvider):
    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        return "/"

    def format_payload(self, prompt, max_tokens):
        data = super().format_payload(prompt, max_tokens)
        data["ignore_eos"] = True
        data["stream_tokens"] = data.pop("stream")
        return data

    def parse_output_json(self, data, prompt):
        if not self.parsed_options.stream:
            data = data["output"]
        return super().parse_output_json(data, prompt)
