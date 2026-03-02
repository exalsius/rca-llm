import logging
import os
import pathlib
from typing import Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import RayPrometheusStatLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")

app = FastAPI()


def download_gguf_file(model_name_or_path: str) -> str:
    # Only proceed if the URL ends with .gguf
    if not model_name_or_path.endswith(".gguf"):
        logger.info("File does not have .gguf suffix, skipping download.")
        return model_name_or_path  # Return original URL if not a .gguf file
    # Define download path
    download_path = pathlib.Path("/tmp/models")
    download_path.mkdir(parents=True, exist_ok=True)
    # Extract file name and define full download path
    file_name = pathlib.Path(model_name_or_path).name
    file_path = download_path.joinpath(file_name)
    repo_id = str(pathlib.Path(model_name_or_path).parent)
    # Download the file if it doesn't already exist
    if not file_path.exists():
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=download_path)
        logger.info(f"Downloaded {file_name} to {file_path}")
    else:
        logger.info(f"{file_name} already exists at {file_path}")
    # Return the new file path
    return str(file_path)


def get_served_model_names(engine_args: AsyncEngineArgs) -> List[str]:
    if engine_args.served_model_name is not None:
        served_model_names: Union[str, List[str]] = engine_args.served_model_name
        # Because the typing suggests it could be a string or list of strings
        if isinstance(served_model_names, str):
            served_model_names: List[str] = [served_model_names]
    else:
        served_model_names: List[str] = [engine_args.model]
    return served_model_names


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: Literal["auto", "string", "openai"] = "auto",
    ):
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        engine_args.model = download_gguf_file(engine_args.model)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.engine_args = engine_args
        self.vllm_config: VllmConfig = self.engine_args.create_engine_config(
            UsageContext.ENGINE_CONTEXT
        )
        # Configure custom logger so that vllm metrics are also exposed
        served_model_names: List[str] = get_served_model_names(self.engine_args)
        additional_metrics_logger: RayPrometheusStatLogger = RayPrometheusStatLogger(
            local_interval=0.5,
            labels=dict(model_name=served_model_names[0]),
            vllm_config=self.vllm_config,
        )
        self.engine.add_logger("ray", additional_metrics_logger)

    def get_serving_models(self) -> OpenAIServingModels:
        served_model_names: List[str] = get_served_model_names(self.engine_args)
        base_model_paths = [
            BaseModelPath(name=name, model_path=self.engine_args.model)
            for name in served_model_names
        ]
        return OpenAIServingModels(
            self.engine,
            self.vllm_config.model_config,
            base_model_paths,
            lora_modules=self.lora_modules,
            prompt_adapters=None,
        )

    @app.post("/completions")
    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_completion:
            model_config = await self.engine.get_model_config()
            serving_models: OpenAIServingModels = self.get_serving_models()
            self.openai_serving_completion = OpenAIServingCompletion(
                self.engine, model_config, serving_models, request_logger=None
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        assert isinstance(generator, CompletionResponse)
        return JSONResponse(content=generator.model_dump())

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            serving_models: OpenAIServingModels = self.get_serving_models()
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                serving_models,
                self.response_role,
                request_logger=None,
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for k, v in cli_args.items():
        if v is not None:
            arg_strings.extend([f"--{k}", str(v)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    pp = engine_args.pipeline_parallel_size
    num_workers = tp * pp  # total vLLM engine replicas

    logger.info(f"Tensor parallelism = {tp}, Pipeline parallelism = {pp}")
    pg_resources = [{"CPU": 1}]  # head node

    cpu_per_actor = int(os.environ["BUILD_APP_ARG_CPU_PER_ACTOR"])
    gpu_per_actor = int(os.environ["BUILD_APP_ARG_GPU_PER_ACTOR"])

    for _ in range(num_workers):
        pg_resources.append(
            {"CPU": cpu_per_actor, "GPU": gpu_per_actor}
        )  # for the vLLM actors

    if num_workers > 1:
        return VLLMDeployment.options(
            placement_group_bundles=pg_resources,
            placement_group_strategy=os.environ.get(
                "BUILD_APP_ARG_PLACEMENT_GROUP_STRATEGY", "PACK"
            ),
        ).bind(
            engine_args,
            parsed_args.response_role,
            parsed_args.lora_modules,
            parsed_args.chat_template,
            parsed_args.chat_template_content_format,
        )
    else:
        return VLLMDeployment.bind(
            engine_args,
            parsed_args.response_role,
            parsed_args.lora_modules,
            parsed_args.chat_template,
            parsed_args.chat_template_content_format,
        )


# Initialize an empty dictionary
dynamic_ray_engine_args = {}
# Iterate over all environment variables
for key, value in os.environ.items():
    # Check if the environment variable starts with the prefix "DYNAMIC_RAY_CLI_ARG"
    if key.startswith("DYNAMIC_RAY_CLI_ARG") and value is not None and len(value):
        # Remove the prefix, convert to lowercase, and replace underscores with hyphens
        processed_key = key[len("DYNAMIC_RAY_CLI_ARG_") :].lower().replace("_", "-")
        # Add the processed key and its value to the dictionary
        dynamic_ray_engine_args[processed_key] = value

model = build_app(dynamic_ray_engine_args)
