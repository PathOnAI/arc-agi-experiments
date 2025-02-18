import asyncio
import hashlib
import json
import os
import random
import string
import time
import traceback
import typing as T
from copy import deepcopy
from enum import Enum
from pathlib import Path

import httpx
import numpy as np
from devtools import debug
from pydantic import BaseModel, computed_field
from tqdm import tqdm

from src import USE_GRID_URL, logfire
from src.db import init_db_pool, pool
from src.prompts import prompts

DOUBLE_ENTER = "\n\n"

GRID = list[list[int]]


# random string
def random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


class Model(str, Enum):
    claude_3_5_sonnet = "claude-3-5-sonnet-20241022"
    claude_3_5_haiku = "claude-3-5-haiku-20241022"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"
    o1_mini = "o1-mini"
    o3_mini = "o3-mini"
    o1_preview = "o1-preview"
    azure_gpt_4o = "azure-gpt-4o"
    azure_gpt_4o_mini = "azure-gpt-4o-mini"
    nvidia_llama_3_1_nemotron_70b_instruct = "nvidia/llama-3.1-nemotron-70b-instruct"
    groq_llama_3_2_90b_vision = "llama-3.2-90b-vision-preview"
    # openrouter_claude_3_5_sonnet = "anthropic/claude-3.5-sonnet"
    openrouter_claude_3_5_sonnet = "anthropic/claude-3.5-sonnet:beta"
    openrouter_o1 = "openai/o1-preview"
    openrouter_o1_mini = "openai/o1-mini-preview"
    # gemini_1_5_pro = "gemini-1.5-pro"
    gemini_1_5_pro = "gemini-1.5-pro-002"
    deep_seek_r1 = "deepseek-reasoner"
    baseten_deepseek_r1 = "baseten_deepseek_r1"


class ModelPrice(BaseModel):
    cache_create_per_million_cents: float
    cache_read_per_million_cents: float
    input_tokens_per_million_cents: float
    output_tokens_per_million_cents: float


model_price_map: dict[Model, ModelPrice] = {
    Model.claude_3_5_sonnet: ModelPrice(
        cache_create_per_million_cents=375,
        cache_read_per_million_cents=0.30,
        input_tokens_per_million_cents=300,
        output_tokens_per_million_cents=1_500,
    ),
    Model.claude_3_5_haiku: ModelPrice(
        cache_create_per_million_cents=125,
        cache_read_per_million_cents=0.1,
        input_tokens_per_million_cents=100,
        output_tokens_per_million_cents=500,
    ),
    Model.gpt_4o: ModelPrice(
        cache_create_per_million_cents=250,
        cache_read_per_million_cents=125,
        input_tokens_per_million_cents=250,
        output_tokens_per_million_cents=1_000,
    ),
    Model.gpt_4o_mini: ModelPrice(
        cache_create_per_million_cents=15,
        cache_read_per_million_cents=7.5,
        input_tokens_per_million_cents=15,
        output_tokens_per_million_cents=60,
    ),
    Model.o1_preview: ModelPrice(
        cache_create_per_million_cents=1500,
        cache_read_per_million_cents=750,
        input_tokens_per_million_cents=1500,
        output_tokens_per_million_cents=6000,
    ),
    Model.o1_mini: ModelPrice(
        cache_create_per_million_cents=300,
        cache_read_per_million_cents=150,
        input_tokens_per_million_cents=300,
        output_tokens_per_million_cents=1200,
    ),
    Model.openrouter_o1: ModelPrice(
        cache_create_per_million_cents=1500,
        cache_read_per_million_cents=1500,
        input_tokens_per_million_cents=1500,
        output_tokens_per_million_cents=6000,
    ),
    Model.openrouter_o1_mini: ModelPrice(
        cache_create_per_million_cents=300,
        cache_read_per_million_cents=300,
        input_tokens_per_million_cents=300,
        output_tokens_per_million_cents=1200,
    ),
    Model.openrouter_claude_3_5_sonnet: ModelPrice(
        cache_create_per_million_cents=300,
        cache_read_per_million_cents=300,
        input_tokens_per_million_cents=300,
        output_tokens_per_million_cents=1_500,
    ),
    Model.gemini_1_5_pro: ModelPrice(
        cache_create_per_million_cents=450,
        cache_read_per_million_cents=0.3125,
        input_tokens_per_million_cents=125,
        output_tokens_per_million_cents=500,
    ),
    Model.deep_seek_r1: ModelPrice(
        cache_create_per_million_cents=55,
        cache_read_per_million_cents=14,
        input_tokens_per_million_cents=55,
        output_tokens_per_million_cents=219,
    ),
    Model.baseten_deepseek_r1: ModelPrice(  # TODO change eventually
        cache_create_per_million_cents=55,
        cache_read_per_million_cents=14,
        input_tokens_per_million_cents=55,
        output_tokens_per_million_cents=219,
    ),
    Model.o3_mini: ModelPrice(
        cache_create_per_million_cents=110,
        cache_read_per_million_cents=55,
        input_tokens_per_million_cents=110,
        output_tokens_per_million_cents=440,
    ),
}


class Example(BaseModel):
    input: GRID
    output: GRID


class Challenge(BaseModel):
    id: str
    train: list[Example]
    test: list[Example]


class Metadata(BaseModel):
    num_tokens_used: int
    latency_ms: float


class FixInfo(BaseModel):
    use_fix_reasoning_tags: bool
    use_if_fix_fail_line: bool
    use_typical_issue_text: bool
    include_diffs: bool


class ModelUsage(BaseModel):
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    input_tokens: int
    output_tokens: int


class KTopConfig(BaseModel):
    k_top: int
    unique_code: bool
    unique_output: bool


class Prompt(str, Enum):
    REASONING = "REASONING"
    COT = "COT"
    ONLY_GRID = "ONLY_GRID"


prompt_map: dict[Prompt, str] = {
    Prompt.REASONING: prompts.REASONING_PROMPT_PYTHON,
    Prompt.COT: prompts.COT_PROMPT_PYTHON,
    Prompt.ONLY_GRID: prompts.DIRECT_GRID_NO_THOUGHTS_PROMPT,
}

prompt_returns_python_map: dict[Prompt, bool] = {
    Prompt.REASONING: True,
    Prompt.COT: True,
    Prompt.ONLY_GRID: False,
}


class PromptConfig(BaseModel):
    base_prompt: Prompt
    use_ascii: bool
    use_array: bool
    use_image: bool

    @computed_field
    @property
    def returns_python(self) -> bool:
        return prompt_returns_python_map[self.base_prompt]


class RootPromptConfig(PromptConfig):
    base_prompt: Prompt
    use_examples: bool
    use_diffs: bool
    use_images: bool


class FixPromptConfig(PromptConfig):
    use_fix_reasoning_tags: bool
    use_fix_fail_line: bool
    use_typical_issue_text: bool
    include_diffs: bool


class LLMConfig(BaseModel):
    model: Model
    temperature: float


class AttemptConfig(BaseModel):
    attempts: int
    llm_config: LLMConfig
    fixes: list["AttemptEdge"]
    include_all_attempts_in_fixes: bool = False


class RootAttemptConfig(AttemptConfig):
    prompt_config: RootPromptConfig


class FixAttemptConfig(AttemptConfig):
    prompt_config: FixPromptConfig


class PoolingConfig(BaseModel):
    size: int


class AttemptEdge(BaseModel):
    k_top_config: KTopConfig
    configs: list[FixAttemptConfig]
    pooling: PoolingConfig | None = None


class Attempt(BaseModel):
    id: str

    config: RootAttemptConfig | FixAttemptConfig
    usage: ModelUsage

    challenge: Challenge
    messages: list[dict[str, T.Any]]

    python_code_str: str | None

    train_attempts: list[GRID]
    test_attempt: GRID

    fixing: "Attempt" = None

    def __hash__(self):
        return hash(self.id)

    @computed_field
    @property
    def fixing_id(self) -> str | None:
        if not self.fixing:
            return None
        return self.fixing.id

    @computed_field
    @property
    def fixing_ids(self) -> list[str]:
        if not self.fixing:
            return []
        return [self.fixing.id, *self.fixing.fixing_ids]

    @computed_field
    @property
    def train_accuracy(self) -> float:
        if not self.challenge.train:
            return 0
        num_correct = 0
        for train_ind in range(len(self.challenge.train)):
            if self.challenge.train[train_ind].output == self.train_attempts[train_ind]:
                num_correct += 1
        return num_correct / len(self.challenge.train)

    @computed_field
    @property
    def avg_cell_diff_percent(self) -> float:
        if not self.challenge.train:
            return 0

        def percent_right_from_grids(train_output: GRID, train_attempt: GRID) -> float:
            try:
                if len(train_output) != len(train_attempt):
                    return 0
                if len(train_output[0]) != len(train_attempt[0]):
                    return 0

                num_right = 0
                rows = len(train_output)
                cols = len(train_output[0])

                for row in range(rows):
                    for col in range(cols):
                        if train_output[row][col] == train_attempt[row][col]:
                            num_right += 1
                return num_right / (rows * cols)
            except Exception as e:
                logfire.debug(f"in percent right from grids: {e=}")
                return 0

        avg_right_lst: list[float] = []

        for train_ind in range(len(self.challenge.train)):
            avg_right_lst.append(
                percent_right_from_grids(
                    train_output=self.challenge.train[train_ind].output,
                    train_attempt=self.train_attempts[train_ind],
                )
            )

        return sum(avg_right_lst) / len(avg_right_lst)

    @computed_field
    @property
    def test_accuracy(self) -> float:
        if self.challenge.test[0].output == self.test_attempt:
            return 1
        return 0

    @staticmethod
    def cost_cents_from_usage(model: Model, usage: ModelUsage) -> float:
        model_price = model_price_map[model]
        return (
            usage.cache_creation_input_tokens
            * model_price.cache_create_per_million_cents
            + usage.cache_read_input_tokens * model_price.cache_read_per_million_cents
            + usage.input_tokens * model_price.input_tokens_per_million_cents
            + usage.output_tokens * model_price.output_tokens_per_million_cents
        ) / 1_000_000

    @computed_field
    @property
    def cost_cents(self) -> float:
        return self.cost_cents_from_usage(
            model=self.config.llm_config.model, usage=self.usage
        )

    @staticmethod
    async def llm_response_to_result_grids(
        challenge: Challenge, llm_response: str, returns_python: bool
    ) -> tuple[str | None, GRID, list[GRID]]:
        from src.llms import parse_2d_arrays_from_string, parse_python_backticks
        from src.run_python import run_python_transform_async

        if returns_python:
            python_str = parse_python_backticks(llm_response)
            # debug(python_str)
            transform_results = await run_python_transform_async(
                code=python_str,
                grid_lists=[
                    deepcopy(challenge.test[0].input),
                    *[deepcopy(train.input) for train in challenge.train],
                ],
                timeout=7,
                raise_exception=True,
            )
            logfire.debug(
                f"Transform results took {transform_results.latency_ms:.2f} ms"
            )
            test_grid = transform_results.transform_results[0]
            train_grids = transform_results.transform_results[1:]
        else:
            python_str = None
            lists = parse_2d_arrays_from_string(s=llm_response)
            if not lists:
                logfire.debug(f"LLM RESPONSE: {llm_response}")
                raise ValueError("No arrays found in output")
            test_grid = lists[-1]
            train_grids = [[[-1, -1], [-1, -1]]] * len(challenge.train)

        return python_str, test_grid, train_grids

    @staticmethod
    async def llm_responses_to_result_grids_list(
        llm_responses: list[str], challenge: Challenge, returns_python: bool
    ) -> list[tuple[str | None, GRID, list[GRID]] | None]:
        result_grids_list: list[tuple[str | None, GRID, list[GRID]] | None] = []

        tasks = [
            Attempt.llm_response_to_result_grids(
                challenge=challenge,
                llm_response=llm_response,
                returns_python=returns_python,
            )
            for llm_response in llm_responses
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logfire.debug(f"FAILED LLM RESPONSE: {r}")
                result_grids_list.append(None)
            else:
                result_grids_list.append(r)

        return result_grids_list

    @classmethod
    async def from_messages_many(
        cls,
        challenge: Challenge,
        messages: list[dict[str, T.Any]],
        attempt_config: RootAttemptConfig | FixAttemptConfig,
        n_times: int,
    ) -> list["Attempt"]:
        from src.llms import get_next_messages

        try:
            next_messages = await get_next_messages(
                messages=deepcopy(messages),
                model=attempt_config.llm_config.model,
                temperature=attempt_config.llm_config.temperature,
                n_times=n_times,
            )
            if not next_messages:
                return []
        except Exception as e:
            logfire.debug(
                f"[{challenge.id}] BIG PROBLEM***** Error getting next messages: {e}"
            )
            return []
        start_grid = time.time()
        llm_responses = [m[0] for m in next_messages]
        grid_lists = None
        if USE_GRID_URL:
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(
                        url="https://arc-agi-grid-306099487123.us-central1.run.app/llm_responses_to_grid_list",
                        json={
                            "llm_responses": llm_responses,
                            "challenge": Challenge.model_dump(challenge, mode="json"),
                        },
                        params={
                            "returns_python": attempt_config.prompt_config.returns_python
                        },
                    )
                    logfire.debug(
                        f"[{challenge.id}] getting grid from server took {r.elapsed.total_seconds()}"
                    )
                grid_lists = r.json()
            except Exception as e:
                logfire.debug(f"ERROR RUNNING GRIDLISTS SERVER: {e}")
                grid_lists = None

        if grid_lists is None:
            grid_lists = await cls.llm_responses_to_result_grids_list(
                llm_responses=llm_responses,
                challenge=challenge,
                returns_python=attempt_config.prompt_config.returns_python,
            )
        logfire.debug(f"[{challenge.id}] grids took {time.time() - start_grid} secs")
        attempts: list[Attempt] = []
        for next_message, grid_list in zip(next_messages, grid_lists, strict=True):
            if grid_list:
                python_str, test_grid, train_grids = grid_list
                llm_response, usage = next_message
                attempts.append(
                    Attempt(
                        id=f"{challenge.id}-{random_string()}",
                        challenge=challenge,
                        messages=[
                            *messages,
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": llm_response}],
                            },
                        ],
                        python_code_str=python_str,
                        train_attempts=train_grids,
                        test_attempt=test_grid,
                        config=attempt_config,
                        usage=usage,
                    )
                )
        return attempts

    @classmethod
    def messages_from_fixes(
        cls,
        challenge: Challenge,
        attempt_config: FixAttemptConfig,
        fixing: list["Attempt"],
        # root_attempt_config: RootAttemptConfig,
    ) -> list[dict[str, T.Any]]:
        from src.logic import challenge_to_messages
        from src.reps import grid_diffs_to_ascii, grid_to_ascii

        assert len(fixing) > 0
        if not isinstance(attempt_config.prompt_config, FixPromptConfig):
            raise Exception("Given prompt must be a fix prompt.")

        class TrainExampleAttempts(BaseModel):
            i: int
            input: GRID
            output: GRID
            wrong_attempts: list[GRID]
            correct_attempts: list[GRID]
            python_codes_for_attempt: list[str]

        train_example_attempts: list[TrainExampleAttempts] = []

        for i, example in enumerate(challenge.train):
            wrong_attempts: list[GRID] = []
            correct_attempts: list[GRID] = []
            python_codes_for_attempt: list[str] = []
            for fix in fixing:
                train_attempt = fix.train_attempts[i]
                python_codes_for_attempt.append(fix.python_code_str)
                if example.output != train_attempt:
                    wrong_attempts.append(train_attempt)
                else:
                    correct_attempts.append(train_attempt)
            train_example_attempts.append(
                TrainExampleAttempts(
                    i=i,
                    input=example.input,
                    output=example.output,
                    wrong_attempts=wrong_attempts,
                    correct_attempts=correct_attempts,
                    python_codes_for_attempt=python_codes_for_attempt,
                )
            )

        attempt_strs: list[str] = []

        # i am including previous attempts to solve this problem and the results of those attempts.
        # learn from these failed attempts to solve this problem.

        for i, fix in enumerate(fixing):
            last_message_content = fix.messages[-1]["content"]
            if type(last_message_content) is not str:
                last_message_content = last_message_content[-1]["text"]
            # debug(last_message_content)
            ss_list: list[str] = []
            for example_i, fix_train_attempt in enumerate(fix.train_attempts):
                challenge_train_input = np.array(challenge.train[example_i].input)
                challenge_train_output = np.array(challenge.train[example_i].output)
                try:
                    fix_train_attempt_np = np.array(fix_train_attempt)
                except Exception as e:
                    logfire.debug(f"FAILED TO CONVERT TO ARRAY: {e}")
                    break

                if challenge_train_output.shape == fix_train_attempt_np.shape:
                    diff_str = (
                        f"## Color changes between the Expected Output ASCII and your Transformed Output:"
                        f"{DOUBLE_ENTER}{grid_diffs_to_ascii(grid_input=challenge_train_output, grid_output=fix_train_attempt_np, separator='|')}"
                    )
                else:
                    diff_str = ""

                try:
                    is_correct = (challenge_train_output == fix_train_attempt_np).all()
                except Exception as e:
                    logfire.debug(f"Error with is correct: {e}")
                    is_correct = False
                if is_correct:
                    incorrect_str = """Your `transform` function was correct on this example! Think about why it worked on this and not others."""
                else:
                    incorrect_str = f"""
# Incorrect Transformed Output ASCII representation:{DOUBLE_ENTER}{grid_to_ascii(grid=fix_train_attempt_np)}

{diff_str.strip()}
                    """

                ss = f"""
# Example {example_i}:

# Input ASCII representation:{DOUBLE_ENTER}{grid_to_ascii(grid=challenge_train_input)}

# Expected Output ASCII representation:{DOUBLE_ENTER}{grid_to_ascii(grid=challenge_train_output)}

{incorrect_str.strip()}
                """
                ss_list.append(ss.strip())
            s = f"""
### Your previous incorrect response #{i}:
<response>
{last_message_content}
</response>

### The results of using your `transform` function on each example:
{DOUBLE_ENTER.join(ss_list)}
                """
            attempt_strs.append(s.strip())

        reasoning_tag_str = (
            "<fix_reasoning></fix_reasoning>"
            if attempt_config.prompt_config.use_fix_reasoning_tags
            else "<reasoning></reasoning>"
        )

        typical_issue_text = "\n\nIf you notice an issue with your previous understanding of the transformation rule, you'll need to do further analysis (including analyzing properties of the example inputs and outputs) to determine exactly what the correct transformation rule is."
        if not attempt_config.prompt_config.use_typical_issue_text:
            typical_issue_text = ""

        if_fix_fail_line = "\n\nIf your attempted fix fails, you'll be called again (in the same way) to continue debugging. So, if print statements would help you debug, you can include them in your code."
        if not attempt_config.prompt_config.use_fix_fail_line:
            if_fix_fail_line = ""

        pool_fix_prompt = f"""
The `transform` function you implemented failed on at least one of the examples you were provided.

Your task is to determine what the issue is and then fix the code. The issue could be a bug in the code and/or an issue with your previous understanding of the transformation rule.

You'll need to carefully reason to determine the issue and to determine how to fix the code. Start your response by doing this reasoning in {reasoning_tag_str} tags. Then, implement the fixed transformation in code.

I have asked you this same question before. I am showing you many of your attempts to solve this problem that were incorrect in some way. I am including your entire response, including your reasoning that led to the wrong `transform` function.

Here are the examples that have failed:

{DOUBLE_ENTER.join(attempt_strs)}

Recall that you should start by reasoning to determine what the issue is in {reasoning_tag_str} tags. Also recall that the problem could be a bug in the code and/or an issue with your previous understanding of the transformation rule.{typical_issue_text}

Once you are done reasoning, rewrite the code to fix the issue. Return the code in triple backticks (```python and then ```).{if_fix_fail_line}
        """.strip()

        messages = challenge_to_messages(
            challenge=challenge,
            add_examples=True,
            include_diffs=True,
            prompt=Prompt.REASONING,
            include_image=True,
            use_ascii=True,
            use_array=True,
        )
        messages.append(deepcopy(fixing[0].messages[-1]))
        messages.append(
            {
                "role": "user",
                "content": [{"text": pool_fix_prompt, "type": "text"}],
            }
        )

        if str(messages).count("cache_control") < 4:
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        return messages

    @classmethod
    async def run_many(
        cls,
        challenge: Challenge,
        attempt_config: RootAttemptConfig | FixAttemptConfig,
        raise_exception: bool,
        fixing: list["Attempt"],
        n_times: int,
    ) -> list["Attempt"]:
        from src.logic import challenge_to_messages

        if not fixing:
            assert isinstance(attempt_config, RootAttemptConfig)
            messages = challenge_to_messages(
                challenge=challenge,
                add_examples=attempt_config.prompt_config.use_examples,
                include_diffs=attempt_config.prompt_config.use_diffs,
                prompt=attempt_config.prompt_config.base_prompt,
                include_image=attempt_config.prompt_config.use_image,
                use_ascii=attempt_config.prompt_config.use_ascii,
                use_array=attempt_config.prompt_config.use_array,
            )
        else:
            assert isinstance(attempt_config, FixAttemptConfig)
            # TODO check if it is already correct, in which case return
            messages = cls.messages_from_fixes(
                challenge=challenge, attempt_config=attempt_config, fixing=fixing
            )
        return await cls.from_messages_many(
            challenge=challenge,
            messages=messages,
            attempt_config=attempt_config,
            n_times=n_times,
        )

    async def fix_many(
        self,
        *,
        attempt_config: FixAttemptConfig,
        return_correct_attempt: bool = True,
        raise_exception: bool,
        n_times: int,
    ) -> list["Attempt"]:
        from src.reps import grid_diffs_to_ascii, grid_to_ascii

        if not isinstance(attempt_config.prompt_config, FixPromptConfig):
            raise Exception("Given prompt must be a fix prompt.")

        wrong_attempts: list = []
        for i, example in enumerate(self.challenge.train):
            train_attempt = self.train_attempts[i]
            if example.output != train_attempt:
                wrong_attempts.append(
                    {
                        "ind": i,
                        "input": example.input,
                        "output": example.output,
                        "attempt": train_attempt,
                    }
                )

        if not wrong_attempts:
            if return_correct_attempt:
                return []
            raise Exception("NO WRONG ATTEMPTS, WOOHOOO")

        wrong_attempt_strs: list[str] = []
        for wrong_attempt in wrong_attempts:
            if attempt_config.prompt_config.include_diffs:
                grid_input_temp = np.array(wrong_attempt["input"])
                try:
                    grid_output_temp = np.array(wrong_attempt["attempt"])
                except Exception as e:
                    logfire.debug(f"FAILED TO CONVERT TO ARRAY: {e}")
                    continue
                if grid_input_temp.shape == grid_output_temp.shape:
                    diff_str = (
                        f"## Color changes between the Input and Output ASCII representation:"
                        f"{DOUBLE_ENTER}{grid_diffs_to_ascii(grid_input=grid_input_temp, grid_output=grid_output_temp, separator='|')}{DOUBLE_ENTER}"
                    )
                else:
                    diff_str = ""
            else:
                diff_str = ""
            s = f"""
# Example {wrong_attempt['ind']}:

# Input ASCII representation:{DOUBLE_ENTER}{grid_to_ascii(grid=np.array(wrong_attempt['input']))}{DOUBLE_ENTER}

# Incorrect Transformed Output ASCII representation:{DOUBLE_ENTER}{grid_to_ascii(grid=np.array(wrong_attempt['attempt']))}{DOUBLE_ENTER}

# Expected Output ASCII representation:{DOUBLE_ENTER}{grid_to_ascii(grid=np.array(wrong_attempt['output']))}{DOUBLE_ENTER}

{diff_str}
                """
            wrong_attempt_strs.append(s.strip())

        wrong_attempt_str = DOUBLE_ENTER.join(wrong_attempt_strs)

        reasoning_tag_str = (
            "<fix_reasoning></fix_reasoning>"
            if attempt_config.prompt_config.use_fix_reasoning_tags
            else "<reasoning></reasoning>"
        )

        if_fix_fail_line = "\n\nIf your attempted fix fails, you'll be called again (in the same way) to continue debugging. So, if print statements would help you debug, you can include them in your code."

        if not attempt_config.prompt_config.use_fix_fail_line:
            if_fix_fail_line = ""

        typical_issue_text = "\n\nIf you notice an issue with your previous understanding of the transformation rule, you'll need to do further analysis (including analyzing properties of the example inputs and outputs) to determine exactly what the correct transformation rule is."

        if not attempt_config.prompt_config.use_typical_issue_text:
            typical_issue_text = ""

        prompt = f"""
The `transform` function you implemented failed on at least one of the examples you were provided. Your task is to determine what this issue is and then fix the code. The issue could be a bug in the code and/or an issue with your previous understanding of the transformation rule.

You'll need to carefully reason to determine the issue and to determine how to fix the code. Start your response by doing this reasoning in {reasoning_tag_str} tags. Then, implement the fixed transformation in code.

Here are the examples that have failed:

{wrong_attempt_str}{DOUBLE_ENTER}

Ok, that is all of the actual and expected outputs.

Recall that you should start by reasoning to determine what the issue is in {reasoning_tag_str} tags. Also recall that the problem could be a bug in the code and/or an issue with your previous understanding of the transformation rule.{typical_issue_text}

Once you are done reasoning, rewrite the code to fix the issue. Return the code in triple backticks (```python and then ```).{if_fix_fail_line}
            """.strip()

        messages = deepcopy(self.messages)
        # don't need cache since this specific fix is only called a few times
        # maybe if it is called more this makes sense
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        )

        attempts = await self.from_messages_many(
            challenge=self.challenge,
            messages=messages,
            attempt_config=attempt_config,
            n_times=n_times,
        )
        for attempt in attempts:
            attempt.fixing = self
        return attempts

    def plot(self, ignore_fixing: bool) -> None:
        from src.plot import plot_results

        if ignore_fixing:
            plot_results([self])
            return

        all_results: list[Attempt] = []
        q: list[Attempt] = [self]

        while q:
            if q in all_results:
                continue
            curr = q.pop()
            if curr.fixing:
                q.append(curr.fixing)
            all_results.append(curr)

        plot_results(all_results[::-1])

    def to_db_query(self, run_id: str) -> tuple[str, list[T.Any]]:
        d = self.model_dump(mode="json")
        s = """
            INSERT INTO attempts(
                id,
                config,
                usage,
                challenge,
                messages,
                python_code_str,
                train_attempts,
                test_attempt,
                fixing_id,
                run_id,
                train_accuracy,
                test_accuracy,
                avg_cell_diff_percent,
                cost_cents,
                fixing_ids
            ) VALUES (
                $1,
                $2::jsonb,
                $3::jsonb,
                $4::jsonb,
                $5::jsonb[],
                $6,
                $7::jsonb[],
                $8::jsonb,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15
            )
            """
        return s, [
            d["id"],
            json.dumps(d["config"]),
            json.dumps(d["usage"]),
            json.dumps(d["challenge"]),
            [json.dumps(m) for m in d["messages"]],
            d["python_code_str"],
            [json.dumps(m) for m in d["train_attempts"]],
            json.dumps(d["test_attempt"]),
            d["fixing_id"],
            run_id,
            d["train_accuracy"],
            d["test_accuracy"],
            d["avg_cell_diff_percent"],
            d["cost_cents"],
            d["fixing_ids"],
        ]

    async def insert(self, run_id: str | None) -> None:
        await self.insert_many(attempts=[self], run_id=run_id)

    @staticmethod
    async def insert_run(run_id: str, started_at_ms: float, ended_at_ms: float) -> None:
        global pool
        if not pool:
            await init_db_pool()
            from src.db import pool

        async with pool.acquire() as conn:
            # Use execute_many for efficient bulk insertion
            await conn.execute(
                "INSERT INTO runs(id, started_at_ms, ended_at_ms) VALUES ($1, $2, $3)",
                run_id,
                started_at_ms,
                ended_at_ms,
            )

    @staticmethod
    async def insert_many(attempts: list["Attempt"], run_id: str | None) -> None:
        """
        Bulk insert multiple attempts into the database.

        Args:
            attempts: List of Attempt objects to insert
            run_id: Optional run ID to associate with all attempts
        """
        global pool
        if not pool:
            await init_db_pool()
            from src.db import pool

        async with pool.acquire() as conn:
            # Create a list of records for bulk insertion
            values_list: list[tuple] = []
            if not attempts:
                return None
            for attempt in attempts:
                s, vals = attempt.to_db_query(run_id=run_id)
                values_list.append(tuple(vals))

            # Use execute_many for efficient bulk insertion
            await conn.executemany(s, values_list)
