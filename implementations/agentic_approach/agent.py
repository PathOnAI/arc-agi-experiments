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

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def write_to_file(file_path: str, text: str, encoding: str = "utf-8") -> str:
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding=encoding) as f:
            f.write(text)
        return "File written successfully."
    except Exception as error:
        return f"Error: {error}"

def run_python_script(script_name):
    try:
        result = subprocess.run(["python", script_name], capture_output=True, text=True, check=True)
        res = f"stdout:{result.stdout}"
        if result.stderr:
            res += f"stderr:{result.stderr}"
        return res
    except subprocess.CalledProcessError as e:
        return f"Error:{e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "description": "Write string content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Full file name with path where the content will be written."
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content to be written into the file."
                    },
                    "encoding": {
                        "type": "string",
                        "default": "utf-8",
                        "description": "Encoding to use for writing the file. Defaults to 'utf-8'."
                    }
                },
                "required": ["file_path", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_script",
            "description": "Execute a Python script in a subprocess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {
                        "type": "string",
                        "description": "The name with path of the script to be executed."
                    }
                },
                "required": ["script_name"]
            }
        }
    }
]

client = OpenAI()
available_tools = {
    "write_to_file": write_to_file,
    "run_python_script": run_python_script,
}

def process_tool_calls(tool_calls):
    tool_responses = []
    logger.info("Number of function calls: %i", len(tool_calls))
    
    for tool_call in tool_calls:
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        function_to_call = available_tools.get(function_name)

        try:
            function_response = function_to_call(**function_args)
            logger.info('function name: %s, function args: %s, function response: %s', 
                       function_name, function_args, function_response)
            
            tool_response = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response)
            }
            tool_responses.append(tool_response)
            
        except Exception as e:
            logger.error(f"Error while calling function {function_name}: {e}")
            tool_responses.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": f"Error: {e}"
            })

    return tool_responses

def send_completion_request(messages: list = None, tools: list = None, depth = 0) -> dict:
    if depth >= 8:
        return None

    if tools is None:
        response = completion(model=MODEL, messages=messages)
        logger.info('depth: %s, response: %s', depth, response)
        message = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
        messages.append(message)
        return response

    response = completion(
        model=MODEL, 
        messages=messages, 
        tools=tools, 
        tool_choice="auto"
    )

    if not response.choices[0].message.tool_calls:
        logger.info('no function calling, depth: %s, response: %s', depth, response)
        message = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
        messages.append(message)
        return response

    logger.info('has function calling, depth: %s, response: %s', depth, response)
    message = {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "tool_calls": response.choices[0].message.tool_calls
    }
    messages.append(message)
    
    tool_responses = process_tool_calls(response.choices[0].message.tool_calls)
    messages.extend(tool_responses)
    return send_completion_request(messages, tools, depth + 1)

def send_prompt(messages, content: str):
    messages.append({"role": "user", "content": content})
    return send_completion_request(messages, tools, 0)



def save_messages_to_json(messages, filename="coding_messages.json"):
    def serialize_message(msg):
        if isinstance(msg, dict):
            serialized_msg = {}
            for k, v in msg.items():
                if k == 'tool_calls' and v is not None:
                    serialized_msg[k] = [
                        {
                            'id': tool_call.id,
                            'type': tool_call.type,
                            'function': {
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments
                            }
                        } for tool_call in v
                    ]
                else:
                    serialized_msg[k] = v
            return serialized_msg
        return str(msg)

    formatted_messages = []
    for index, message in enumerate(messages):
        formatted_message = {
            "index": index,
            "message": serialize_message(message),
            "type": str(type(message))
        }
        formatted_messages.append(formatted_message)

    with open(filename, 'w') as f:
        json.dump(formatted_messages, f, indent=2)
    print(f"Messages saved to {filename}")
# Initialize messages
messages = [{
    "role": "system", 
    "content": "You are a coding agent, you first write code per instruction, then write test case, and run the test, if there is bug, debug it"
}]

# Example usage
send_prompt(messages, "the problem is Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.You can return the answer in any order.")
save_messages_to_json(messages, filename="messages.json")