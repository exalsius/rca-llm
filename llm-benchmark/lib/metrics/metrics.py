import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional

from locust import events

logger = logging.getLogger("llm-benchmark")


def add_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="METRIC",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )


@dataclass
class ChunkMetadata:
    text: str
    logprob_tokens: Optional[int]
    usage_tokens: Optional[int]
    prompt_usage_tokens: Optional[int]


class FixedQPSPacer:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, qps, distribution):
        self.qps = qps
        self.distribution = distribution

        # It's kind of thread safe thanks to GIL as the only state is `t` - good enough for a loadtest
        def gen():
            t = time.time()
            mean_wait = 1 / self.qps
            while True:
                if self.distribution == "exponential":
                    wait = random.expovariate(1 / mean_wait)
                elif self.distribution == "uniform":
                    wait = random.uniform(0, 2 * mean_wait)
                elif self.distribution == "constant":
                    wait = mean_wait
                else:
                    logger.warning("Unknown distribution {self.distribution}")
                    os._exit(1)
                t += wait
                yield t

        self.iterator = gen()

    @classmethod
    def instance(cls, qps, distribution):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(qps, distribution)
            else:
                assert cls._instance.qps == qps
                assert cls._instance.distribution == distribution
            return cls._instance

    def wait_time_till_next(self):
        with self._lock:
            t = next(self.iterator)
        now = time.time()
        if now > t:
            logger.warning(
                f"WARNING: not enough locust users to keep up with the desired QPS. Either the number of locust users is too low or the server is overloaded. Delay: {now-t:.3f}s"
            )
            return 0
        return t - now


class LengthSampler:
    def __init__(self, distribution: str, mean: int, cap: Optional[int], alpha: float):
        self.distribution = distribution
        self.mean = mean
        self.cap = cap
        self.alpha = alpha

        if self.distribution == "exponential":
            self.sample_func = lambda: int(random.expovariate(1 / self.mean))
        elif self.distribution == "uniform":
            mx = self.mean + int(self.alpha * self.mean)
            if self.cap is not None:
                mx = min(mx, self.cap)
            self.sample_func = lambda: random.randint(
                max(1, self.mean - int(self.alpha * self.mean)), mx
            )
        elif self.distribution == "constant":
            self.sample_func = lambda: self.mean
        elif self.distribution == "normal":
            self.sample_func = lambda: int(
                random.gauss(self.mean, self.mean * self.alpha)
            )
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

    def sample(self) -> int:
        for _ in range(1000):
            sample = self.sample_func()
            if sample <= 0:
                continue
            if self.cap is not None and sample > self.cap:
                continue
            return sample
        else:
            raise ValueError(
                "Can't sample a value after 1000 attempts, check distribution parameters"
            )

    def __str__(self):
        r = int(self.mean * self.alpha)
        if self.distribution == "constant":
            s = str(self.mean)
        elif self.distribution == "uniform":
            s = f"uniform({self.mean} +/- {r})"
        elif self.distribution == "normal":
            s = f"normal({self.mean}, {r})"
        elif self.distribution == "exponential":
            s = f"exponential({self.mean})"
        else:
            assert False
        if self.cap is not None:
            s += f" capped at {self.cap}"
        return s


class InitTracker:
    lock = threading.Lock()
    users = None
    first_request_done = 0
    logging_params = None
    environment = None
    tokenizer = None

    @classmethod
    def notify_init(cls, environment, logging_params):
        with cls.lock:
            if cls.environment is None:
                cls.environment = environment
            if cls.logging_params is None:
                cls.logging_params = logging_params
            else:
                assert (
                    cls.logging_params == logging_params
                ), f"Inconsistent settings between workers: {cls.logging_params} != {logging_params}"

    @classmethod
    def notify_first_request(cls):
        with cls.lock:
            if (
                cls.environment.parsed_options.qps is not None
                and cls.first_request_done == 0
            ):
                # if in QPS mode, reset after first successful request comes back
                cls.reset_stats()
            cls.first_request_done += 1
            if (
                cls.environment.parsed_options.qps is not None
                and cls.first_request_done == 0
                and cls.users == cls.first_request_done
            ):
                # if in fixed load mode, reset after all users issued one request (we're in a steady state)
                cls.reset_stats()

    @classmethod
    def notify_spawning_complete(cls, user_count):
        with cls.lock:
            cls.users = user_count
            if cls.users == cls.first_request_done:
                cls.reset_stats()

    @classmethod
    def reset_stats(cls):
        assert cls.environment.runner, "only local mode is supported"
        logger.info("Resetting stats after traffic reach a steady state")
        cls.environment.events.reset_stats.fire()
        cls.environment.runner.stats.reset_all()

    @classmethod
    def load_tokenizer(cls, dir):
        if not dir:
            return None
        with cls.lock:
            if cls.tokenizer:
                return cls.tokenizer
            import transformers

            cls.tokenizer = transformers.AutoTokenizer.from_pretrained(dir)
            cls.tokenizer.add_bos_token = False
            cls.tokenizer.add_eos_token = False
            return cls.tokenizer


events.spawning_complete.add_listener(InitTracker.notify_spawning_complete)
