import asyncio
from alpine.client.client import LLMClient

async def main():
    client = LLMClient()
    messages = [{"role":"user", "content": "what's up?"}]
    async for event in client.chat_Completion(messages, False):
        print(event)
    print('done')

def cli():
    asyncio.run(main())
