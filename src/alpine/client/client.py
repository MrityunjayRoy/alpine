from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from typing import Any
from collections.abc import AsyncGenerator

import asyncio
import os
from dotenv import load_dotenv

from alpine.client.response import StreamEvent, TextDelta, TokenUsage, StreamEventType


load_dotenv()

class LLMClient:
    def __init__(self) -> None: 
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                        base_url="https://openrouter.ai/api/v1/",
                        api_key=os.getenv("CLIENT_API_KEY")
                    )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_Completion(
            self,
            messages: list[dict[str, Any]],
            stream: bool = True
            ) -> AsyncGenerator[StreamEvent, None]:

        client = self.get_client() 
        kwargs = {
                "model": "nvidia/nemotron-3.5-super-vl:free",
                "messages": messages,
                "stream": stream,
                }

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event

                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                            type=StreamEventType.ERROR,
                            error = "Rate limit exceeded!!"
                            )

                    return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                            type=StreamEventType.ERROR,
                            error = "Error connecting to API"
                            )

                    return

    async def _stream_response(
            self,
            client: AsyncOpenAI,
            kwargs: dict[str, Any],
            ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        usage: TokenUsage | None = None
        finish_reason: str | None = None
        
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                        )
            
            if not chunk.choices:
                continue

            choices = chunk.choices[0]
            delta = choices.delta

            if choices.finish_reason:
                finish_reason = choices.finish_reason

            if delta.content:
                yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=TextDelta(delta.content),
                        )

        yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason=finish_reason,
                usage=usage,
                )

    async def _non_stream_response(
            self,
            client: AsyncOpenAI,
            kwargs: dict[str, Any],
            ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choices = response.choices[0]
        message = choices.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
                    )

        return StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                text_delta=text_delta,
                finish_reason=None,
                usage=usage
                )
