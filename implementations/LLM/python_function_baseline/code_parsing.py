import re
from typing import Optional

class CodeBlockParser:
    """Parser for extracting and cleaning Python code blocks from text."""
    
    DEFAULT_CODE = """
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError()
""".strip()

    def __init__(self, tab_size: int = 4):
        self.tab_size = tab_size
        
    def clean_whitespace(self, code: str) -> str:
        """Normalize whitespace by converting tabs to spaces."""
        return code.replace("\t", " " * self.tab_size)

    def extract_transform_function(self, text: str) -> str:
        """
        Extract the transform function from text containing Python code blocks.
        Returns the DEFAULT_CODE if no valid code block is found.
        """
        # Handle case with no code blocks
        if "```python" not in text:
            content = text.partition("</reasoning>")[2]
            return self.clean_whitespace(content) if content else self.DEFAULT_CODE

        # Find all Python code blocks
        code_blocks = text.split("```python")
        
        # If multiple blocks exist, find the one with transform function
        if len(code_blocks) > 1:
            for block in reversed(code_blocks):
                if "def transform(" in block:
                    return self._parse_code_block("```python" + block)
        
        # Handle single code block
        return self._parse_code_block(text)

    def _parse_code_block(self, text: str) -> str:
        """Parse a single code block using different patterns."""
        patterns = [
            r"```python\n(.*)\n```",  # Standard pattern
            r"```python\n(.*)\n`",    # Alternative ending
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                return self.clean_whitespace(match.group(1))
        
        # Fallback: try to extract everything after ```python
        remaining_code = text.partition("```python")[2]
        return self.clean_whitespace(remaining_code) if remaining_code else self.DEFAULT_CODE

# Example usage:
def parse_python_code(text: str) -> str:
    """
    Main interface function for parsing Python code blocks.
    Returns cleaned code with normalized whitespace.
    """
    parser = CodeBlockParser()
    return parser.extract_transform_function(text)