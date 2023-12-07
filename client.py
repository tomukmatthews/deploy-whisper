"""
See api/api_v2/whisper/README.md for example usage.
"""

from __future__ import annotations

import asyncio
import logging

from abc import ABC, abstractmethod
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Awaitable, cast, Iterator, Literal

from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
from bentoml.client import AsyncClient, Client, SyncClient

# from data_loader.media_file_metadata.media_file_metadata import MediaFileMetadata
from decouple import config

from deployment.service import WHISPER_TRANSCRIBE_ENDPOINT_NAME, WhisperResults
from faster_whisper.audio import decode_audio
from loguru import logger

# from utils.timing import Timer

WHISPER_URI = cast(str, config("WHISPER_URI", default="http://localhost:3000"))
WHISPER_READY_TIMEOUT = cast(
    int, config("WHISPER_READY_TIMEOUT", cast=int, default=120)
)
DEFAULT_SAMPLE_RATE_HZ = 16000

WhisperPredictionInput = str | Path
ExtractedAudio = npt.NDArray[np.float32]
SampleRate = Literal[16000, 44100, 48000]


class _LogInterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Intercept the standard logging message and redirect it to loguru
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


async def gather_with_concurrency(
    *tasks: Awaitable[Any],
    max_concurrent_tasks: int,
    progress_bar: bool = False,
    **tqdm_kwargs,
) -> list[Any]:
    """
    Run tasks concurrently while limiting the maximum number of concurrent tasks.

    Args:
        max_concurrent_tasks (int): Maximum number of concurrent tasks to run.
        *tasks (async function): One or more async function(s) to run concurrently.

    Returns:
        list: The result of all tasks when they are complete.

    Usage:
        async def task(num):
            await asyncio.sleep(1)
            return num * num

        tasks = [task(i) for i in range(10)]
        results = asyncio.run(gather_with_concurrency(*tasks, max_concurrent_tasks=3))
        print(results)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    """
    from tqdm.asyncio import tqdm_asyncio

    semaphore = asyncio.BoundedSemaphore(max_concurrent_tasks)

    async def task_with_semaphore(async_task):
        async with semaphore:
            return await async_task

    if progress_bar:
        return await tqdm_asyncio.gather(
            *(task_with_semaphore(task) for task in tasks), **tqdm_kwargs
        )
    return await asyncio.gather(*(task_with_semaphore(task) for task in tasks))


class WhisperBaseClient(ABC):
    API_PROTOCOL = "http"
    SCHEME = f"{API_PROTOCOL}://"

    def __init__(
        self,
        client: Client,
        whisper_uri: str = WHISPER_URI,
        sample_rate_hz: SampleRate = DEFAULT_SAMPLE_RATE_HZ,
    ):
        self.whisper_uri = whisper_uri
        self._sample_rate_hz = sample_rate_hz
        self.client = client
        self.patch_bentoml_logging()

    def patch_bentoml_logging(self):
        """BentoML logs a warning when loading a pydantic model from a URL for every request. This is not a problem so filter
        out the warning to avoid unnecessary noise in the logs.

        This is a temporary fix until the issue is resolved: https://github.com/bentoml/BentoML/issues/3276#issuecomment-1829288847
        """

        def bentoml_filter(record):
            unwanted_message = "BentoML does not support loading pydantic models from URLs; output will be a normal dictionary."
            return unwanted_message not in record.getMessage()

        bentoml_logger = logging.getLogger("bentoml")
        # Remove existing handlers from the BentoML logger to prevent duplicate logs
        bentoml_logger.handlers = []
        intercept_handler = _LogInterceptHandler()
        intercept_handler.addFilter(bentoml_filter)
        bentoml_logger.addHandler(intercept_handler)
        bentoml_logger.setLevel(logging.DEBUG)

    @staticmethod
    def extract_audio_from_file(
        file: Path | str, sample_rate_hz: int
    ) -> ExtractedAudio | None:
        try:
            audio_array = decode_audio(
                str(file), sampling_rate=sample_rate_hz, split_stereo=False
            )
            return audio_array
        except Exception as e:
            logger.exception(f"Failed to load audio: {e}")
            return None

    @singledispatchmethod
    async def extract_audio(
        self, file: WhisperPredictionInput, sample_rate_hz: int
    ) -> ExtractedAudio | None:
        return self.extract_audio_from_file(file=file, sample_rate_hz=sample_rate_hz)


class WhisperSyncClient(WhisperBaseClient):
    """
    Basic Usage:
        file = "video.mp4"
        with WhisperSyncClient() as whisper_client:
            result = whisper_client.predict(file)
    """

    def __init__(
        self,
        client: SyncClient | None = None,
        whisper_uri: str = WHISPER_URI,
        sample_rate_hz: SampleRate = DEFAULT_SAMPLE_RATE_HZ,
    ):
        super().__init__(
            client=client, whisper_uri=whisper_uri, sample_rate_hz=sample_rate_hz
        )
        if client is None:
            self.client = SyncClient.from_url(whisper_uri, kind=self.API_PROTOCOL)
        self._wait_until_server_ready()

    def _wait_until_server_ready(self):
        parsed_uri = urlparse(self.whisper_uri, scheme=WhisperBaseClient.SCHEME)
        self.client.wait_until_server_ready(
            host=parsed_uri.hostname,
            port=parsed_uri.port,
            timeout=WHISPER_READY_TIMEOUT,
        )

    def predict(
        self, file: WhisperPredictionInput, *, job_id: str | None = None
    ) -> WhisperResults:
        audio_array = asyncio.run(self.extract_audio(file, self._sample_rate_hz))

        if audio_array is None:
            return WhisperResults(segments=[], info=None)

        whisper_result = self.client.call(
            bentoml_api_name=WHISPER_TRANSCRIBE_ENDPOINT_NAME, inp=audio_array
        )
        whisper_result = WhisperResults(**whisper_result)

        return whisper_result

    def close(self):
        self.client.close()

    def __enter__(self) -> WhisperSyncClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


class WhisperAsyncClient(WhisperBaseClient):
    """
    Basic Usage:
        async with await WhisperAsyncClient.create() as whisper_client:
            result = await whisper_client.predict(file)

    Batch processing example:
        import asyncio

        from deployment.service import WhisperResults
        from client import WhisperAsyncClient

        async def run_whisper_async(video_files: list[str]) -> list[WhisperResults]:
            async with await WhisperAsyncClient.create() as whisper_client:
                tasks = [whisper_client.predict(S3URI(file)) for file in video_files]
                results = await gather_with_concurrency(
                    *tasks, max_concurrent_tasks=1, progress_bar=True
                )
                return results


        video_files = ["example_videos/ForBiggerMeltdowns.mp4"] * 10
        results = asyncio.run(run_whisper_async(video_files))
    """

    def __init__(
        self,
        client: AsyncClient,
        whisper_uri: str = WHISPER_URI,
        sample_rate_hz: SampleRate = DEFAULT_SAMPLE_RATE_HZ,
    ):
        super().__init__(
            client=client, whisper_uri=whisper_uri, sample_rate_hz=sample_rate_hz
        )

    async def predict(
        self, file: WhisperPredictionInput, *, job_id: str | None = None
    ) -> WhisperResults:
        """
        Usage:
            whisper_client = await WhisperAsyncClient.create()
            result = await whisper_client.predict(s3_uri)
        """

        audio_array = await self.extract_audio(file, self._sample_rate_hz)

        if audio_array is None:
            return WhisperResults(segments=[], info=None)

        whisper_result = await self.client.call(
            bentoml_api_name=WHISPER_TRANSCRIBE_ENDPOINT_NAME, inp=audio_array
        )

        whisper_result = WhisperResults(**whisper_result)

        return whisper_result

    @staticmethod
    async def _wait_until_server_ready(client: AsyncClient, server_uri: str):
        parsed_uri = urlparse(server_uri, scheme=WhisperBaseClient.SCHEME)
        await client.wait_until_server_ready(
            host=parsed_uri.hostname,
            port=parsed_uri.port,
            timeout=WHISPER_READY_TIMEOUT,
        )

    @classmethod
    async def create(
        cls: type[WhisperAsyncClient],
        whisper_uri: str = WHISPER_URI,
        sample_rate_hz: SampleRate = DEFAULT_SAMPLE_RATE_HZ,
    ) -> WhisperAsyncClient:
        client = await AsyncClient.from_url(
            whisper_uri, kind=WhisperBaseClient.API_PROTOCOL
        )
        await WhisperAsyncClient._wait_until_server_ready(client, whisper_uri)
        return WhisperAsyncClient(
            client=client,
            whisper_uri=whisper_uri,
            sample_rate_hz=sample_rate_hz,
        )

    async def close(self):
        await self.client.close()

    async def __aenter__(self) -> WhisperAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()


if __name__ == "__main__":
    from deployment.service import WhisperResults

    async def run_whisper_async(video_files: list[str]) -> list[WhisperResults]:
        async with await WhisperAsyncClient.create() as whisper_client:
            tasks = [whisper_client.predict(file) for file in video_files]
            results = await gather_with_concurrency(
                *tasks, max_concurrent_tasks=1, progress_bar=True
            )
            return results

    video_files = ["example_videos/ForBiggerMeltdowns.mp4"] * 10
    results = asyncio.run(run_whisper_async(video_files))
