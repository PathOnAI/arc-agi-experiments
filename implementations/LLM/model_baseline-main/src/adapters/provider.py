import abc
from typing import List, Dict, Tuple
import json

class ProviderAdapter(abc.ABC):
    @abc.abstractmethod
    def chat_completion(self, message: str) -> str:
        pass

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        pass