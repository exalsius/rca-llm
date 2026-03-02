from lib.provider.fireworks_provider import FireworksProvider
from lib.provider.openai_provider import OpenAIProvider
from lib.provider.tgi_provider import TgiProvider
from lib.provider.together_provider import TogetherProvider
from lib.provider.triton_generate_provider import TritonGenerateProvider
from lib.provider.triton_infer_provider import TritonInferProvider
from lib.provider.vllm_provider import VllmProvider

PROVIDER_CLASS_MAP = {
    "fireworks": FireworksProvider,
    "vllm": VllmProvider,
    "openai": OpenAIProvider,
    "anyscale": OpenAIProvider,
    "together": TogetherProvider,
    "triton-infer": TritonInferProvider,
    "triton-generate": TritonGenerateProvider,
    "tgi": TgiProvider,
}
