import os

from devtools import debug
from openai import AsyncOpenAI

from src.llms import get_next_messages
from src.models import Model


async def main() -> None:
    deepseek_client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    res = await deepseek_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "what is 2+2?"},
        ],
        model="deepseek/deepseek-r1:free",
    )

    m = await get_next_messages(
        model=Model.deep_seek_r1,
        temperature=0.95,
        n_times=1,
    )
    debug(m)


async def main_again() -> None:
    # Please install OpenAI SDK first: `pip3 install openai`

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False,
    )
    debug(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_again())
