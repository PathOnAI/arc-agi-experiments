import os

from devtools import debug
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! How are you?"},
]
message = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, max_tokens=100
)
debug(message.choices[0].message.content)
