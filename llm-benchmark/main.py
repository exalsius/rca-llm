import argparse
import copy
import csv
import json
import logging
import random
import sys
import time
import traceback
from functools import partial

from lib import PROVIDER_CLASS_MAP
from lib.metrics.metrics import (
    FixedQPSPacer,
    InitTracker,
    LengthSampler,
    add_custom_metric,
)
from lib.utils.utils import _load_curl_like_data
from locust import HttpUser, constant_pacing, events, task
from orjson import orjson

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Ensure logs go to stdout
)
logger = logging.getLogger("llm-benchmark")

prompt_prefix = "Pad "  # exactly one token
# "Lengthy" prompt borrowed from nat.dev
prompt = """Generate a Django application with Authentication, JWT, Tests, DB support. Show docker-compose for python and postgres. Show the complete code for every file!"""
prompt_tokens = 35  # from Llama tokenizer tool (so we don't import it here)
prompt_random_tokens = 10

events.spawning_complete.add_listener(InitTracker.notify_spawning_complete)


class LLMUser(HttpUser):
    # no wait time, so every user creates a continuous load, sending requests as quickly as possible

    def on_start(self):
        try:
            self._on_start()
        except Exception as e:
            logger.error(f"Failed to initialize: {repr(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)

    def _guess_provider(self):
        self.model = self.environment.parsed_options.model
        self.provider = self.environment.parsed_options.provider
        # guess based on URL
        if self.provider is None:
            if "fireworks.ai" in self.host:
                self.provider = "fireworks"
            elif "together" in self.host:
                self.provider = "together"
            elif "openai" in self.host:
                self.provider = "openai"
            elif "anyscale" in self.host:
                self.provider = "anyscale"

        if (
            self.model is None
            and self.provider is not None
            and PROVIDER_CLASS_MAP[self.provider].DEFAULT_MODEL_NAME is not None
        ):
            self.model = PROVIDER_CLASS_MAP[self.provider].DEFAULT_MODEL_NAME

        if self.model and self.provider:
            return

        # vllm doesn't support /model/<name> endpoint, so iterate over all models
        try:
            resp = self.client.get("/v1/models")
            resp.raise_for_status()
            resp = resp.json()
        except Exception as e:
            raise ValueError(
                "Argument --model or --provider was not specified and /v1/models failed"
            ) from e

        models = resp["data"]
        assert len(models) > 0, "No models found in /v1/models"
        owned_by = None
        # pick the first model
        for m in models:
            if self.model is None or m["id"] == self.model:
                self.model = m["id"]
                owned_by = m["owned_by"]
                break
        if self.provider is None:
            if not owned_by:
                raise ValueError(
                    f"Model {self.model} not found in /v1/models. Specify --provider explicitly"
                )
            if owned_by in ["vllm", "fireworks"]:
                self.provider = owned_by
            else:
                raise ValueError(
                    f"Can't detect provider, specify it explicitly with --provider, owned_by={owned_by}"
                )

    def _on_start(self):
        self.client.headers["Content-Type"] = "application/json"
        if self.environment.parsed_options.api_key:
            self.client.headers["Authorization"] = (
                "Bearer " + self.environment.parsed_options.api_key
            )
        self._guess_provider()
        logger.info(
            f" Provider {self.provider} using model {self.model} ".center(80, "*")
        )
        self.provider_formatter = PROVIDER_CLASS_MAP[self.provider](
            self.model, self.environment.parsed_options
        )

        self.stream = self.environment.parsed_options.stream
        prompt_chars = self.environment.parsed_options.prompt_chars
        if self.environment.parsed_options.prompt_text:
            self.prompt = _load_curl_like_data(
                self.environment.parsed_options.prompt_text
            )
        elif prompt_chars:
            self.prompt = (
                prompt_prefix * (prompt_chars // len(prompt_prefix) + 1) + prompt
            )[:prompt_chars]
        else:
            min_prompt_len = (
                prompt_tokens
                + prompt_random_tokens
                * self.environment.parsed_options.prompt_randomize
            )
            assert (
                self.environment.parsed_options.prompt_tokens >= min_prompt_len
            ), f"Minimal prompt length is {min_prompt_len}"
            self.prompt = (
                prompt_prefix
                * (self.environment.parsed_options.prompt_tokens - min_prompt_len)
                + prompt
            )
        self.max_tokens_sampler = LengthSampler(
            distribution=self.environment.parsed_options.max_tokens_distribution,
            mean=self.environment.parsed_options.max_tokens,
            cap=self.environment.parsed_options.max_tokens_cap,
            alpha=self.environment.parsed_options.max_tokens_range,
        )
        self.temperature = self.environment.parsed_options.temperature

        logging_params = {
            # TODO: add some server info with git version
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.environment.parsed_options.prompt_tokens,  # might be overwritten based on metric
            "generation_tokens": str(self.max_tokens_sampler),
            "stream": self.stream,
            "temperature": self.temperature,
            "logprobs": self.environment.parsed_options.logprobs,
        }
        InitTracker.notify_init(self.environment, logging_params)

        self.tokenizer = InitTracker.load_tokenizer(
            self.environment.parsed_options.tokenizer
        )
        if self.tokenizer:
            self.prompt_tokenizer_tokens = len(
                self.tokenizer.encode(self._get_prompt())
            )
        else:
            self.prompt_tokenizer_tokens = None

        if self.environment.parsed_options.qps is not None:
            if self.environment.parsed_options.burst:
                raise ValueError("Burst and QPS modes are mutually exclusive")
            pacer = FixedQPSPacer.instance(
                self.environment.parsed_options.qps,
                self.environment.parsed_options.qps_distribution,
            )
            # it will be called by Locust after each task
            self.wait_time = pacer.wait_time_till_next
            self.wait()
        elif self.environment.parsed_options.burst:
            self.wait_time = partial(
                constant_pacing(self.environment.parsed_options.burst), self
            )
        else:
            # introduce initial delay to avoid all users hitting the service at the same time
            time.sleep(random.random())

        self.first_done = False

    def _get_prompt(self):
        if not self.environment.parsed_options.prompt_randomize:
            return self.prompt
        # single letters are single tokens
        return (
            " ".join(
                chr(ord("a") + random.randint(0, 25))
                for _ in range(prompt_random_tokens)
            )
            + " "
            + self.prompt
        )

    @task
    def generate_text(self):
        max_tokens = self.max_tokens_sampler.sample()
        prompt = self._get_prompt()
        data = self.provider_formatter.format_payload(prompt, max_tokens)
        t_start = time.perf_counter()

        with self.client.post(
            self.provider_formatter.get_url(),
            data=json.dumps(data),
            stream=True,
            catch_response=True,
        ) as response:
            dur_chunks = []
            combined_text = ""
            done = False
            prompt_usage_tokens = self.prompt_tokenizer_tokens
            total_usage_tokens = None
            total_logprob_tokens = None
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e
            t_first_token = None
            for chunk in response.iter_lines(delimiter=b"\n"):
                if len(chunk.strip()) == 0:
                    continue  # come providers send empty lines between data chunks
                if t_first_token is None:
                    t_prev = time.perf_counter()
                if done:
                    if chunk != b"data: [DONE]":
                        logger.warning(
                            f"WARNING: Received more chunks after [DONE]: {chunk}"
                        )
                try:
                    now = time.perf_counter()
                    dur_chunks.append(now - t_prev)
                    t_prev = now
                    if self.stream:
                        assert chunk.startswith(
                            b"data:"
                        ), f"Unexpected chunk not starting with 'data': {chunk}"
                        chunk = chunk[len(b"data:") :]
                        if chunk.strip() == b"[DONE]":
                            done = True
                            continue
                    data = orjson.loads(chunk)
                    out = self.provider_formatter.parse_output_json(data, prompt)
                    if out.usage_tokens:
                        total_usage_tokens = (
                            total_usage_tokens or 0
                        ) + out.usage_tokens
                    if out.prompt_usage_tokens:
                        prompt_usage_tokens = out.prompt_usage_tokens
                    if out.text and t_first_token is None:
                        logger.info(f"First token received: {out.text}")
                        t_first_token = time.perf_counter()
                    combined_text += out.text

                    if out.logprob_tokens:
                        total_logprob_tokens = (
                            total_logprob_tokens or 0
                        ) + out.logprob_tokens
                except Exception as e:
                    logger.error(
                        f"Failed to parse response: {chunk} with error {repr(e)}"
                    )
                    response.failure(e)
                    sys.exit(1)
                    return
            assert t_first_token is not None, "empty response received"
            if (
                (total_logprob_tokens is not None)
                and (total_usage_tokens is not None)
                and total_logprob_tokens != total_usage_tokens
            ):
                logger.warning(
                    f"WARNING: usage_tokens {total_usage_tokens} != logprob_tokens {total_logprob_tokens}"
                )
            if total_logprob_tokens is not None:
                num_tokens = total_logprob_tokens
            else:
                num_tokens = total_usage_tokens
            if self.tokenizer:
                num_tokenizer_tokens = len(self.tokenizer.encode(combined_text))
                if num_tokens is None:
                    num_tokens = num_tokenizer_tokens
                elif num_tokens != num_tokenizer_tokens:
                    logger.warning(
                        f"WARNING: tokenizer token count {num_tokenizer_tokens} != {num_tokens} received from server"
                    )
                    num_tokens = num_tokenizer_tokens
            num_tokens = num_tokens or 0
            num_chars = len(combined_text)
            now = time.perf_counter()
            dur_total = now - t_start
            dur_generation = now - t_first_token
            dur_first_token = t_first_token - t_start
            logger.info(
                f"Response received: total {dur_total*1000:.2f} ms, first token {dur_first_token*1000:.2f} ms, {num_chars} chars, {num_tokens} tokens"
            )
            if self.environment.parsed_options.show_response:
                logger.info("---")
                logger.info(combined_text)
                logger.info("---")
            if num_chars:
                add_custom_metric(
                    "latency_per_char", dur_generation / num_chars * 1000, num_chars
                )
            add_custom_metric("time_to_first_token", dur_first_token * 1000)
            add_custom_metric("total_latency", dur_total * 1000)
            if num_tokens:
                if num_tokens != max_tokens:
                    logger.warning(
                        f"WARNING: wrong number of tokens: {num_tokens}, expected {max_tokens}"
                    )
                add_custom_metric("num_tokens", num_tokens)
                add_custom_metric(
                    "latency_per_token", dur_generation / num_tokens * 1000, num_tokens
                )
                add_custom_metric(
                    "overall_latency_per_token",
                    dur_total / num_tokens * 1000,
                    num_tokens,
                )
            if (
                prompt_usage_tokens is not None
                and self.prompt_tokenizer_tokens is not None
                and prompt_usage_tokens != self.prompt_tokenizer_tokens
            ):
                logger.warning(
                    f"WARNING: prompt usage tokens {prompt_usage_tokens} != {self.prompt_tokenizer_tokens} derived from local tokenizer"
                )
            prompt_tokens = prompt_usage_tokens or self.prompt_tokenizer_tokens
            if prompt_tokens:
                add_custom_metric("prompt_tokens", prompt_tokens)

            if not self.first_done:
                self.first_done = True
                InitTracker.notify_first_request()


@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CLASS_MAP.keys()),
        type=str,
        help="Which flavor of API to use. If not specified, we'll try to guess based on the URL and /v1/models output",
    )
    parser.add_argument(
        "-m",
        "--model",
        env_var="MODEL",
        type=str,
        help="The model to use for generating text. If not specified we will pick the first model from the service as returned by /v1/models",
    )
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /v1/chat/completions API",
    )
    parser.add_argument(
        "-p",
        "--prompt-tokens",
        env_var="PROMPT_TOKENS",
        type=int,
        default=512,
        help="Length of the prompt in tokens. Default 512",
    )
    parser.add_argument(
        "--prompt-chars",
        env_var="PROMPT_CHARS",
        type=int,
        help="Length of the prompt in characters.",
    )
    parser.add_argument(
        "--prompt-text",
        env_var="PROMPT_TEXT",
        type=str,
        help="Prompt text to use instead of generating one. It can be a file reference starting with an ampersand, e.g. `@prompt.txt`",
    )
    parser.add_argument(
        "--prompt-randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include a few random numbers in the generated prompt to avoid caching",
    )
    parser.add_argument(
        "-o",
        "--max-tokens",
        env_var="MAX_TOKENS",
        type=int,
        default=64,
        help="Max number of tokens to generate. If --max-tokens-distribution is non-constant this is going to be the mean. Defaults to 64",
    )
    parser.add_argument(
        "--max-tokens-cap",
        env_var="MAX_TOKENS_CAP",
        type=int,
        help="If --max-tokens-distribution is non-constant, this truncates the distribition at the specified limit",
    )
    parser.add_argument(
        "--max-tokens-distribution",
        env_var="MAX_TOKENS_DISTRIBUTION",
        type=str,
        choices=["constant", "uniform", "exponential", "normal"],
        default="constant",
        help="How to sample `max-tokens` on each request",
    )
    parser.add_argument(
        "--max-tokens-range",
        env_var="MAX_TOKENS_RANGE",
        type=float,
        default=0.3,
        help="Specifies the width of the distribution. Specified value `alpha` is relative to `max-tokens`. For uniform distribution we'd sample from [max_tokens - max_tokens * alpha, max_tokens + max_tokens * alpha]. For normal distribution we'd sample from `N(max_tokens, max_tokens * alpha)`. Defaults to 0.3",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the streaming API",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        env_var="API_KEY",
        help="Auth for the API",
    )
    parser.add_argument(
        "--temperature",
        env_var="TEMPERATURE",
        type=float,
        default=1.0,
        help="Temperature parameter for the API",
    )
    parser.add_argument(
        "--logprobs",
        type=str,
        default=None,
        help="Whether to ask for logprobs. Must be a number (e.g. '5') for completions endpoint and boolean (e.g. 'true') if chat.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        help="Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, writes out the header first",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Enabled 'fixed QPS' mode where requests are issues at the specified rate regardless of how long the processing takes. In this case --users and --spawn-rate need to be set to a sufficiently high value (e.g. 100)",
    )
    parser.add_argument(
        "--qps-distribution",
        type=str,
        choices=["constant", "uniform", "exponential"],
        default="constant",
        help="Must be used with --qps. Specifies how to space out requests: equally ('constant') or by sampling wait times from a distribution ('uniform' or 'exponential'). Expected QPS is going to match --qps",
    )
    parser.add_argument(
        "--burst",
        type=float,
        default=None,
        help="Makes requests to arrive in bursts every specified number of seconds. Note that burst duration has to be longer than maximum time of the response. Size of the burst is controlled by --users. The spawn rate -r is best set to a high value",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Specify HF tokenizer to use for validating the output of the model. It's optional, we're going to rely on 'usage' or 'logprobs' field to get token count information",
    )
    parser.add_argument(
        "--show-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the result of each generation",
    )


@events.quitting.add_listener
def _(environment, **kw):
    total_latency = environment.stats.entries[("total_latency", "METRIC")]
    if environment.stats.total.num_failures > 0 or total_latency.num_requests == 0:
        logger.error("Test failed due to failed requests")
        environment.process_exit_code = 1
        return

    entries = copy.copy(InitTracker.logging_params)
    if environment.parsed_options.qps is not None:
        entries[
            "concurrency"
        ] = f"QPS {environment.parsed_options.qps} {environment.parsed_options.qps_distribution}"
    else:
        entries["concurrency"] = InitTracker.users
    for metric_name in [
        "time_to_first_token",
        "latency_per_token",
        "num_tokens",
        "total_latency",
        "prompt_tokens",  # might overwrite the static value based on server side tokenization
    ]:
        entries[metric_name] = environment.stats.entries[
            (metric_name, "METRIC")
        ].avg_response_time
    if not environment.parsed_options.stream:
        # if there's no streaming these metrics are meaningless
        entries["time_to_first_token"] = ""
        entries["latency_per_token"] = ""
    entries["num_requests"] = total_latency.num_requests
    entries["qps"] = total_latency.total_rps

    def pretty_name(s: str) -> str:
        return " ".join([w.capitalize() for w in s.split("_")])

    entries = {pretty_name(k): v for k, v in entries.items()}

    # print in the final event handler to make sure our output is the last one
    @events.quit.add_listener
    def exit_printer(**kw):
        max_width = max(len(k) for k in entries.keys())
        logger.info(" Summary ".center(80, "="))
        for k, v in entries.items():
            logger.info(f"{k:<{max_width}}: {v}")
        logger.info("=" * 80)

    if environment.parsed_options.summary_file:
        with open(environment.parsed_options.summary_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=entries.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(entries)
