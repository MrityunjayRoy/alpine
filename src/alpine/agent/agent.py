from collections.abc import AsyncGenerator

from alpine.agent.events import AgentEvent, AgentEventType
from alpine.client.client import LLMClient
from alpine.client.response import StreamEventType


class Agent:
    def __init__(self) -> None:
        self.client = LLMClient()

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        # add user message to context

        final_response_text: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response_text = event.data.get("content")

        yield AgentEvent.agent_end(final_response_text)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        message = [{"role":"system", "content":"hey what are you doing?"}]
        response_text: str = ""

        async for event in self.client.chat_Completion(message, True):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or 'some error occurred')

        if response_text:
            yield AgentEvent.text_complete(content=response_text)

    async def __aenter__(self) -> Agent:
        return self
    
    async def __aexit__(self, exc_t, exc_v, exc_tb) -> None:
        if self.client:
            await self.client.close()
            self.client = None
        
