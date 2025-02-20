from .provider import ProviderAdapter
from src.models import ARCTaskOutput
import anthropic
import os
from dotenv import load_dotenv
import json
from typing import List
load_dotenv()

class AnthropicAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        # Initialize VertexAI model
        self.model = self.init_model()
        self.model_name = model_name
        self.max_tokens = max_tokens

    def init_model(self):
        """
        Initialize the Anthropic model
        """
        
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        return client
    
    def make_prediction(self, prompt: str) -> str:
        """
        Make a prediction with the Anthropic model
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages)

        return response.content[0].text

    def chat_completion(self, messages, tools=[]) -> str:
        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tools
        )
        return response

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        tools = [
            {
                "name": "extract_json",
                "description": "Extracts JSON from the response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            },
                            "description": "A list of lists of integers extracted from the response."
                        }
                    },
                    "required": ["response"]
                }
            }
        ]

        text = f"Extract JSON of the test output from the following response: {input_response}"

        query = f"""
        <document>
        {text}
        </document>

        Use the extract_json tool.
        """

        response = self.chat_completion(
            messages=[{"role": "user", "content": query}],
            tools=tools
        )

        json_response = None
        for content in response.content:
            if content.type == "tool_use" and content.name == "extract_json":
                json_entities = content.input
                break

        if json_entities:
            return json_entities['response']
        else:
            return None
        
if __name__ == "__main__":
    adapter = AnthropicAdapter("claude-3-5-sonnet-20240620")
    print(type(adapter.extract_json_from_response("[[1, 2, 3], [4, 5, 6]]")))