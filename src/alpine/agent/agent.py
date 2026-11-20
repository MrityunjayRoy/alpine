from collections.abc import AsyncGenerator
from pathlib import Path

from alpine.agent.events import AgentEvent, AgentEventType
from alpine.client.client import LLMClient
from alpine.client.response import StreamEventType, ToolCall, ToolResultMessage
from alpine.context.manager import ContextManager
from alpine.tools.registry import create_default_registry


class Agent:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.context_manager = ContextManager()
        self.tool_registry = create_default_registry()

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(content=message)

        final_response_text: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response_text = event.data.get("content")

        yield AgentEvent.agent_end(final_response_text)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        response_text: str = ""

        tool_schemas = self.tool_registry.get_schemas()

        tool_calls: list[ToolCall] = []

        async for event in self.client.chat_Completion(
                self.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                ):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
            elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                if event.tool_call:
                    tool_calls.append(event.tool_call)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or 'some error occurred')

        self.context_manager.add_agent_message(response_text)

        if response_text:
            yield AgentEvent.text_complete(content=response_text)

        tool_call_results: list[ToolResultMessage] = []
        for tool_call in tool_calls:
            yield AgentEvent.tool_call_start(
                call_id=tool_call.call_id,
                name=tool_call.name,
                args=tool_call.arguments,
            )

            result = await self.tool_registry.invoke(
                name=tool_call.name or "",
                params=tool_call.arguments,
                cwd=Path.cwd(),
            )

            yield AgentEvent.tool_call_complete(
                tool_call.call_id,
                tool_call.name,
                result,
            )

            tool_call_results.append(
                ToolResultMessage(
                    tool_call.call_id,
                    content=result.to_model_output(),
                    is_error=not result.success
                )
            )

        for tool_result in tool_call_results:
            self.context_manager.add_tool_result(
                tool_result.tool_call_id,
                tool_result.content
            )

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb) -> None:
        if self.client:
            await self.client.close()
            self.client = None
