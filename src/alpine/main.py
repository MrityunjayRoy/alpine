import asyncio
from alpine.client.client import LLMClient

async def main():
    client = LLMClient()
    messages = [{"role":"user", "content": "what's up?"}]
    async for event in client.chat_Completion(messages, True):
        print(event)
    print('done')

def cli():
    asyncio.run(main())
