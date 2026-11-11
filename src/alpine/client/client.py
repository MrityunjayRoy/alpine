from openai import AsyncOpenAI
from typing import Any

from alpine.client.response import StreamEvent, TextDelta, TokenUsage, EventType

class LLMClient:
    def __init__(self) -> None: 
        self._client : AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                        base_url="https://openrouter.ai/api/v1/",
                        api_key=""
                    )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_Completion(self, messages: list[dict[str, Any]], stream: bool = True):
        client = self.get_client() 
        kwargs = {
                "model": "nvidia/nemotron-3.5-super-vl:free",
                "messages": messages,
                "stream": stream,
                }

        if stream:
            await self._stream_response()
        else:
            await self._non_stream_response(client, kwargs)

    async def _stream_response(self):
        pass

    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any],):
        response = await client.chat.completions.create(**kwargs)
        choices = response.choices[0]
        message = choices.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_token,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    cached_tokens=response.prompt_token_details.cached_tokens,
                    )

        return StreamEvent(
                type=EventType.MESSAGE_COMPLETE,
                text_delta=text_delta,
                error=None,
                finish_reason=None,
                usage=usage
                )
