from openai import OpenAI
# Connect to LM Studio
client = OpenAI(
    base_url="http://localhost:1235/v1",
    api_key="lm-studio"
)

# Send a prompt
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain Python in simple terms."
        }
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
