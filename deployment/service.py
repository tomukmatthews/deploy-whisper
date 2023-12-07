import logging
import typing as t

import bentoml
import numpy as np
import numpy.typing as npt
from faster_whisper import WhisperModel

if t.TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        transcribe: RunnerMethod


import os
from time import time

from decouple import config

from faster_whisper.transcribe import Segment, TranscriptionInfo
from faster_whisper.vad import VadOptions
from loguru import logger
from pydantic import BaseModel

WHISPER_TRANSCRIBE_ENDPOINT_NAME = "transcribe"
WHISPER_MODEL = "small"
WHISPER_MODEL_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
WHISPER_MODEL_KWARGS = dict(
    model_size_or_path=WHISPER_MODEL,
    device="cuda",
    compute_type=config("WHISPER_MODEL_COMPUTE_TYPE", cast=str, default="int8_float16"),
    cpu_threads=config("WHISPER_MODEL_CPU_THREADS", cast=int, default=1),
    num_workers=config("WHISPER_MODEL_NUM_WORKERS", cast=int, default=1),
    download_root=WHISPER_MODEL_CACHE_DIR,
)


class WhisperResults(BaseModel):
    segments: list[Segment]
    info: TranscriptionInfo | None

    @property
    def empty(self) -> bool:
        return len(self.segments) == 0


class _LogInterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Intercept the standard logging message and redirect it to loguru
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


bentoml_logger = logging.getLogger("bentoml")
# Remove existing handlers from the BentoML logger to prevent duplicate logs
bentoml_logger.handlers = []
bentoml_logger.addHandler(_LogInterceptHandler())
bentoml_logger.setLevel(logging.DEBUG)


class WhisperRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = "nvidia.com/gpu"
    SUPPORTS_CPU_MULTI_THREADING = config(
        "SUPPORTS_CPU_MULTI_THREADING", cast=bool, default=False
    )

    def __init__(self):
        self.whisper_model = WhisperModel(**WHISPER_MODEL_KWARGS, local_files_only=True)
        self.inference_kwargs = dict(
            vad_filter=True,
            beam_size=5,
            temperature=0.0,
            # if we leave `max_speech_duration_s` as the default `float("inf")` we can't serialise with json
            vad_parameters=VadOptions(max_speech_duration_s=1e12),
        )
        self.warmup()

    @bentoml.Runnable.method(batchable=False)
    def transcribe(self, audio_array: npt.NDArray) -> WhisperResults:
        logger.info(f"Calling transcribe on shape {audio_array.shape}")
        t0 = time()
        segments, transcription_info = self.whisper_model.transcribe(
            audio_array, **self.inference_kwargs
        )
        t1 = time()
        logger.info(f"Model Transcribed in {t1 - t0:.2f} seconds")
        return WhisperResults(segments=list(segments), info=transcription_info)

    def warmup(self):
        logger.info("Warming up whisper model runner")
        warmup_sample_audio = np.random.rand(10000).astype(np.float32)
        self.whisper_model.transcribe(warmup_sample_audio, **self.inference_kwargs)
        logger.info("Whisper model runner warmed up")


whisper_runner = t.cast(
    "RunnerImpl", bentoml.Runner(WhisperRunnable, name="whisper", embedded=True)
)
svc = bentoml.Service("whisper", runners=[whisper_runner])


@svc.on_startup
def setup_logging(ctx: bentoml.Context):
    bentoml_logger = logging.getLogger("bentoml")
    # Remove existing handlers from the BentoML logger to prevent duplicate logs
    bentoml_logger.handlers = []
    bentoml_logger.addHandler(_LogInterceptHandler())
    bentoml_logger.setLevel(logging.DEBUG)


@svc.api(
    name=WHISPER_TRANSCRIBE_ENDPOINT_NAME,
    input=bentoml.io.NumpyNdarray(dtype=np.float32),
    output=bentoml.io.JSON(pydantic_model=WhisperResults),
)
def transcribe(model_input: npt.NDArray) -> WhisperResults:
    t0 = time()
    result = whisper_runner.transcribe.run(model_input)
    t1 = time()
    logger.info(f"API Transcribed in {t1 - t0:.2f} seconds")
    return result
