import asyncio
from typing import Any
import click
from alpine.client.client import LLMClient

async def run(messages: dict[str, Any]):
    client = LLMClient()
    async for event in client.chat_Completion(messages, True):
        print(event)

@click.command()
@click.argument("prompt", required=False)
def main(
        prompt: str | None
        ):
    print(prompt)
    messages = [{"role":"user", "content": prompt}]
    asyncio.run(run(messages))
    print('done')

main()
