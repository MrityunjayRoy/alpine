from collections.abc import AsyncGenerator

from alpine.agent.events import AgentEvent
from alpine.client.client import LLMClient


class Agent:
    def __init__(self) -> None:
        self.client = LLMClient()

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
       pass 
