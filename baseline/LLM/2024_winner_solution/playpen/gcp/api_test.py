import os

import google.generativeai as genai
from devtools import debug
from dotenv import load_dotenv

from playpen.azure.test_api import messages

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


async def main() -> None:
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = await model.generate_content_async(
        contents=[{"role": "user", "content": "Write a story about a magic backpack."}],
    )
    debug(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
