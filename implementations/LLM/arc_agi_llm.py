import logging
from dotenv import load_dotenv
from litellm import completion
import subprocess
import requests
import os
import json

_ = load_dotenv()
MODEL = "gpt-4o"
MODEL = "claude-3-5-sonnet-20240620"
MODEL = "o3-mini"

# include deep seek r1
# include openai o3-mini

# https://github.com/arcprizeorg/model_baseline/blob/main/prompt_example_o3.md


## step 1:
## add a data loader to transform an input idx to openai messages
## add a function to evaluate the output of the llm output to calculate the accuracy
## step 2:
## loop over the data loader, and call the llm


# Initialize messages
messages = [{
    "role": "system", 
    "content": "You are a coding agent, you first write code per instruction, then write test case, and run the test, if there is bug, debug it"
}]

messages.append({"role": "user", "content": 'calculate 1+1'})
response = completion(
        model=MODEL, 
        messages=messages
    )

print(response)