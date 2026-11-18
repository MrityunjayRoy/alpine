from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from typing import Any
from collections.abc import AsyncGenerator

import asyncio
import os
from dotenv import load_dotenv

from alpine.client.response import StreamEvent, TextDelta, TokenUsage, StreamEventType, ToolCall, ToolCallDelta, parse_tool_call_arguments


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

    def _build_tools(self, tools: list[dict[str, Any]]):
        return [
                {
                    "type": "function",
                    "function": {
                        "name": tool['name'],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    },
                } for tool in tools
                ]

    async def chat_Completion(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            stream: bool = True
            ) -> AsyncGenerator[StreamEvent, None]:

        client = self.get_client() 
        kwargs = {
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "messages": messages,
                "stream": stream,
                }

        if tools:
            kwargs['tools'] = self._build_tools(tools)

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
        tool_calls: dict[int, dict[str, Any]] = {}
        
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

            
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                            "started": False,
                        }

                    tc = tool_calls[idx]

                    if tool_call_delta.id:
                        tc["id"] = tool_call_delta.id

                    function = tool_call_delta.function

                    if function:
                        if function.name:
                            tc["name"] += function.name

                        if tc["name"] and not tc["started"]:
                            tc["started"] = True

                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tc["id"],
                                    name=tc["name"],
                                ),
                            )

                        if function.arguments:
                            arguments = function.arguments
                            tc["arguments"] += arguments

                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tc["id"],
                                    name=tc["name"],
                                    arguments_delta=arguments,
                                ),
                            )

        for idx, tc in tool_calls.items():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=tc["id"],
                    name=tc["name"],
                    arguments=parse_tool_call_arguments(
                        tc["arguments"]
                    ),
                ),
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

        tool_calls: list[ToolCall] = []
        if message.tool_call:
            for tc in message.tool_call:
                tool_calls.append(ToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=parse_tool_call_arguments(tc.function.arguments),
                    )
                )

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
                tool_calls=tool_calls,
                finish_reason=None,
                usage=usage
                )
