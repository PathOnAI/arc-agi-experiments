import logging
from dotenv import load_dotenv
from openai import OpenAI
from litellm import completion
import subprocess
import requests
import os
import json

_ = load_dotenv()
MODEL = "gpt-4o"
MODEL = "claude-3-5-sonnet-20240620"
MODEL = "o3-mini"

## define the function calling tools
## data loader