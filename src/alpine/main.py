import asyncio
from alpine.client.client import LLMClient

async def main():
    client = LLMClient()
    messages = [{"role":"user", "content": "what's up?"}]
    await client.chat_Completion(messages, False)
    print('done')

asyncio.run(main())
