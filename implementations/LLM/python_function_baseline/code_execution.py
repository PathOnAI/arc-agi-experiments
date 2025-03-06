import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from typing import List, Optional, Dict, Any, TypeVar, Union

# Type alias for grid
GRID = List[List[int]]

class PythonException(Exception):
    """Custom exception for Python transform execution errors."""
    pass

class PythonResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        timed_out: bool,
        latency_ms: float,
        transform_results: Optional[List[GRID]]
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.timed_out = timed_out
        self.latency_ms = latency_ms
        self._transform_results = transform_results
        
    @property
    def transform_results(self) -> Optional[List[GRID]]:
        """Get the transformed results as Python objects."""
        if not self._transform_results:
            # Try to parse from stdout if not already parsed
            try:
                for line in self.stdout.splitlines():
                    if line.startswith("TRANSFORM_RESULT:"):
                        results_str = line.replace("TRANSFORM_RESULT:", "", 1)
                        self._transform_results = json.loads(results_str)
                        break
            except (json.JSONDecodeError, AttributeError):
                pass
        return self._transform_results
        
    def get_result(self, index: int = 0) -> Optional[GRID]:
        """Get a specific result grid by index."""
        results = self.transform_results
        if results and 0 <= index < len(results):
            return results[index]
        return None

class PythonTransformExecutor:
    """Executes Python transform functions in a subprocess with timeout."""

    TRANSFORM_RESULT_PREFIX = "TRANSFORM_RESULT:"

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def _create_wrapped_code(self, transform_code: str, grid_lists: List[GRID]) -> str:
        """Create the complete Python code to execute with necessary imports and error handling."""
        return f"""
import json
import sys
import numpy as np
import scipy
from typing import List, Tuple, Set, Union, Optional

# Original transform function
{transform_code}

def to_python_array(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return to_python_array(obj.tolist())
    elif isinstance(obj, list):
        return [to_python_array(item) for item in obj]
    return obj

def main():
    try:
        grid_lists = {json.dumps(grid_lists)}
        results = []
        
        for grid_list in grid_lists:
            result = transform(grid_list)
            result = to_python_array(result)
            
            if not isinstance(result, list) or not all(isinstance(row, list) for row in result):
                print("Error: transform must return List[List[int]]", file=sys.stderr)
                sys.exit(1)
            
            results.append(result)
        
        print("{self.TRANSFORM_RESULT_PREFIX}" + json.dumps(results))
        
    except Exception as e:
        print(f"Error executing transform: {{str(e)}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

    def execute_transform(
        self,
        code: str,
        grid_lists: List[GRID],
        raise_exception: bool = True
    ) -> PythonResult:
        """
        Execute a Python transform function with the provided grid lists.
        
        Args:
            code: Python code containing the transform function
            grid_lists: List of input grids to transform
            raise_exception: Whether to raise an exception on error
            
        Returns:
            PythonResult containing execution results and transformed grids
        """
        start_time = time.time()
        wrapped_code = self._create_wrapped_code(code, grid_lists)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped_code)
            temp_file = f.name

        try:
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            transform_results = None
            timed_out = False

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return_code = process.returncode

                if return_code == 0 and stdout:
                    for line in stdout.splitlines():
                        if line.startswith(self.TRANSFORM_RESULT_PREFIX):
                            try:
                                transform_results = json.loads(
                                    line.replace(self.TRANSFORM_RESULT_PREFIX, "", 1)
                                )
                            except json.JSONDecodeError:
                                stderr = "Error: Could not parse transform result"
                                return_code = 1

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                stderr = f"Execution timed out after {self.timeout} seconds"
                return_code = -1
                timed_out = True

            latency_ms = (time.time() - start_time) * 1000

            if not transform_results and raise_exception:
                raise PythonException(stderr)

            return PythonResult(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                timed_out=timed_out,
                latency_ms=latency_ms,
                transform_results=transform_results
            )

        finally:
            os.unlink(temp_file)

def run_python_transform_sync(
    code: str,
    grid_lists: List[GRID],
    timeout: int = 5,
    raise_exception: bool = True
) -> PythonResult:
    """Convenience function to execute a Python transform."""
    executor = PythonTransformExecutor(timeout=timeout)
    return executor.execute_transform(code, grid_lists, raise_exception)

def run_transforms(input_grid, code: str) -> PythonResult:
    return run_python_transform_sync(
        code=code,
        grid_lists=input_grid,
        timeout=5,
        raise_exception=True
    )