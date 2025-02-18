import json
import os
import time
import traceback
import typing as T
from copy import deepcopy
from enum import Enum

import httpx
import numpy as np
import redis.asyncio as redis
from devtools import debug
from pydantic import BaseModel, TypeAdapter
from tqdm.asyncio import tqdm_asyncio

from src import PLOT, logfire
from src.data import training_challenges
from src.models import (
    GRID,
    Attempt,
    AttemptEdge,
    Challenge,
    FixAttemptConfig,
    Prompt,
    RootAttemptConfig,
    prompt_map,
    random_string,
)
from src.prompts.examples import (
    GRID_CHANGE_PROMPT_EXAMPLE_1,
    GRID_SAME_PROMPT_EXAMPLE_1,
    example_1_grid_change_challenge_id,
    example_1_reasoning_grid_change,
    example_1_same_grid_challenge_id,
    example_1_same_grid_reasoning,
    example_2_challenge_id,
    example_2_reasoning_grid_same,
    example_3_challenge_id,
    example_3_reasoning_grid_same,
    example_7_grid_change_challenge_id,
    example_7_reasoning_grid_change_bad_colors,
)
from src.render_legacy import grid_to_base64_png_oai_content
from src.reps import array_to_str, grid_diffs_to_ascii, grid_to_ascii
from src.run_python import run_python_transform_sync


class TqdmLogfire:
    """File-like class redirecting tqdm progress bar to given logging logger."""

    def __init__(self):
        pass

    def write(self, msg: str) -> None:
        logfire.debug(msg.lstrip("\r"))

    def flush(self) -> None:
        pass


def chunk_list(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def content_blocks_from_matrix(
    *,
    matrix: GRID,
    _label: str,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    matrix = deepcopy(matrix)
    grid = np.array(matrix)
    x, y = grid.shape
    messages = [
        {"type": "text", "text": _label},
        {"type": "text", "text": f"Shape: {x} by {y}\n\n"},
    ]
    if include_image:
        messages.append(grid_to_base64_png_oai_content(grid=grid))
    if use_ascii:
        messages.append(
            {
                "type": "text",
                "text": f"ASCII representation:\n\n{grid_to_ascii(grid=grid, separator='|', spreadsheet_ascii=False)}\n\n",
            }
        )
    if use_array:
        messages.append({"type": "text", "text": array_to_str(grid=matrix)})
    return messages


def content_from_challenge(
    challenge: Challenge,
    include_diffs: bool,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, T.Any]]:
    content = []
    for i, train in enumerate(challenge.train):
        example_number = i + 1
        # add input blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.input,
                _label=f"# Example {example_number}\n\n## Input {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        # add output blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.output,
                _label=f"## Output {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        if not does_grid_change_shape(challenge=challenge) and include_diffs:
            content.append(
                {
                    "type": "text",
                    "text": f"## Color changes between the Input and Output ASCII representation:\n\n"
                    f"{grid_diffs_to_ascii(grid_input=np.array(train.input), grid_output=np.array(train.output), separator='|')}\n\n",
                }
            )

    # TODO for now, only do the first test... Will have to treat these multi tests as multiple examples later
    # assert len(challenge.test) == 1
    content.extend(
        content_blocks_from_matrix(
            matrix=challenge.test[0].input,
            _label="# Additional input\n\n",
            include_image=include_image,
            use_ascii=use_ascii,
            use_array=use_array,
        )
    )

    return content


def does_grid_change_shape(challenge: Challenge) -> bool:
    for train in challenge.train:
        if np.array(train.input).shape != np.array(train.output).shape:
            return True
    return False


def challenge_to_messages(
    *,
    challenge: Challenge,
    add_examples: bool,
    use_cache_control: bool = True,
    include_diffs: bool,
    prompt: Prompt,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    # first, is example same grid size?
    grid_change_shape = does_grid_change_shape(challenge)
    # debug(grid_change_shape)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_map[prompt]}]}
    ]
    if add_examples:
        if grid_change_shape:
            # messages.extend(GRID_CHANGE_PROMPT_EXAMPLE_1)
            example_1_grid_change_prompt = content_from_challenge(
                challenge=training_challenges[example_1_grid_change_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            example_7_grid_change_prompt = content_from_challenge(
                challenge=training_challenges[example_7_grid_change_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example_1_grid_change_prompt,
                    },
                    {"role": "assistant", "content": example_1_reasoning_grid_change},
                ]
            )
        else:
            example_1_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_1_same_grid_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )

            # ADDING OTHER EXAMPLE!
            example_2_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_2_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            example_3_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_3_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example_1_grid_same_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": example_1_same_grid_reasoning,
                    },
                ]
            )

        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Great work! Now I will give you another puzzle to solve just like that one.",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Great, please give me the next puzzle.",
                        }
                    ],
                },
            ]
        )
    if use_cache_control:
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    content = content_from_challenge(
        challenge=challenge,
        include_diffs=include_diffs,
        include_image=include_image,
        use_ascii=use_ascii,
        use_array=use_array,
    )
    if use_cache_control:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    messages.append({"role": "user", "content": content})
    return messages


def eval_attempts(
    attempts: list[Attempt],
    config: RootAttemptConfig | FixAttemptConfig,
    plot: bool,
    time_took_ms: float,
) -> None:
    if not attempts:
        return None

    for attempt in attempts:
        # debug(attempt.train_accuracy, attempt.test_accuracy)
        if plot:
            try:
                start = time.time()
                attempt.plot(ignore_fixing=True)
                took = time.time() - start
                if took > 0.5:
                    logfire.debug(f"TOOK {took} SECONDS TO PLOT")
            except Exception as e:
                logfire.debug(f"FAILED TO PLOT: {e}")

    # get total accuracies
    avg_train_accuracy = sum(attempt.train_accuracy for attempt in attempts) / len(
        attempts
    )
    avg_test_accuracy = sum(attempt.test_accuracy for attempt in attempts) / len(
        attempts
    )
    total_cost = sum(attempt.cost_cents for attempt in attempts)
    total_runs = len(attempts)
    total_correct = len(
        [a for a in attempts if a.test_accuracy == 1 and a.train_accuracy == 1]
    )
    debug_d = {
        "challenge_id": attempts[0].challenge.id,
        "total_runs": total_runs,
        "total_correct": total_correct,
        "avg_train_accuracy": avg_train_accuracy,
        "avg_test_accuracy": avg_test_accuracy,
        "total_cost": total_cost,
        "prompt_config": config.prompt_config,
        "llm_config": config.llm_config,
        "time_took_ms": round(time_took_ms, 2),
    }
    logfire.debug("eval", **debug_d)
    print(
        f"[{attempts[0].challenge.id}] finished processing node [{attempts[0].config.llm_config.model.value}]: {total_runs} attempts, {round(avg_train_accuracy * 100, 2)}% accuracy, ${round(total_cost / 100, 2)}, {round(time_took_ms / 1000, 2)} secs",
    )


def get_best_attempts(
    attempts: list[Attempt], k_top: int, unique_code: bool, unique_output: bool
) -> list[Attempt]:
    # first, order attempts by how many examples they got right
    # then, order by the diff in cells
    # use a better metric later
    example_correct: list[Attempt] = []
    example_wrong: list[Attempt] = []
    for a in attempts:
        if a.train_accuracy > 0:
            example_correct.append(a)
        else:
            example_wrong.append(a)
    sorted_correct = sorted(
        example_correct, key=lambda a: a.train_accuracy, reverse=True
    )
    sorted_wrong = sorted(
        example_wrong,
        key=lambda a: a.avg_cell_diff_percent,
        reverse=True,
    )
    all_sorted: list[Attempt] = [*sorted_correct, *sorted_wrong]

    if unique_code:
        has_seen_python: set[str] = set()
        unique_sorted = []
        for item in all_sorted:
            code_str = item.python_code_str
            if code_str not in has_seen_python:
                unique_sorted.append(item)
                has_seen_python.add(code_str)
        all_sorted = unique_sorted
    if unique_output:
        has_seen_grid: set[str] = set()
        unique_sorted = []
        for item in all_sorted:
            output_grid = str(item.test_attempt)
            if output_grid not in has_seen_grid:
                unique_sorted.append(item)
                has_seen_grid.add(output_grid)
        all_sorted = unique_sorted

    return all_sorted[:k_top]


def get_diverse_attempts(
    root_attempt: Attempt, sorted_attempts: list[Attempt], limit: int
) -> list[Attempt]:
    if root_attempt in sorted_attempts:
        sorted_attempts.remove(root_attempt)
    attempts_by_correct_examples: dict[int, list[Attempt]] = {}
    correct_examples_by_attempt: dict[Attempt, set[int]] = {}
    for a in [root_attempt, *sorted_attempts]:
        for i, train_example in enumerate(a.challenge.train):
            if a.train_attempts[i] == train_example.output:
                if i not in attempts_by_correct_examples:
                    attempts_by_correct_examples[i] = []
                attempts_by_correct_examples[i].append(a)
                if a not in correct_examples_by_attempt:
                    correct_examples_by_attempt[a] = set()
                correct_examples_by_attempt[a].add(i)
    # make sure you have at least one attempt for each correct example
    final_attempts: list[Attempt] = [root_attempt, *sorted_attempts][0:limit]
    count: dict[int, int] = {}
    for a in final_attempts:
        for ii in correct_examples_by_attempt.get(a, set()):
            count[ii] = 1
    # find the missing ones
    missing = attempts_by_correct_examples.keys() - count.keys()
    for miss in missing:
        final_attempts.append(attempts_by_correct_examples[miss][0])
    return final_attempts


def has_perfect_attempts(attempts: list[Attempt]) -> bool:
    attempts_perfect = [a for a in attempts if a.train_accuracy == 1]
    n_perfect = len(attempts_perfect)
    if n_perfect >= 2:
        message = f"[{attempts_perfect[0].challenge.id}] found {n_perfect} solutions from {len(attempts)} attempts"
        logfire.debug(message)
        print(message)
        return True
    return False


async def run_fixes_tree(
    parent_attempts: list[Attempt],
    edges: list[AttemptEdge],
    warm_cache: bool,  # too complex rn w speed
) -> list[Attempt]:
    # DFS fixes
    all_attempts: list[Attempt] = []
    if not parent_attempts:
        challenge_id = ""
    else:
        challenge_id = parent_attempts[0].challenge.id
    if not edges:
        return all_attempts
    for edge in edges:
        best_k = get_best_attempts(
            attempts=parent_attempts,
            k_top=edge.k_top_config.k_top,
            unique_code=edge.k_top_config.unique_code,
            unique_output=edge.k_top_config.unique_output,
        )
        if not best_k:
            continue
        for fix_attempt_config in edge.configs:
            start_level = time.time()
            message = f"[{best_k[0].challenge.id}] running fix node with {fix_attempt_config.attempts * len(best_k)} total attempts."
            print(message)
            logfire.debug(message)
            if fix_attempt_config.attempts == 0:
                continue
            local_attempts: list[Attempt] = []
            tasks = []
            for parent_attempt in best_k:
                if not edge.pooling:
                    tasks.append(
                        parent_attempt.fix_many(
                            attempt_config=fix_attempt_config.model_copy(deep=True),
                            raise_exception=False,
                            n_times=fix_attempt_config.attempts,
                        )
                    )
                else:
                    attempts_to_use = get_diverse_attempts(
                        root_attempt=parent_attempt,
                        sorted_attempts=get_best_attempts(
                            attempts=parent_attempts,
                            k_top=100_000,
                            unique_code=edge.k_top_config.unique_code,
                            unique_output=edge.k_top_config.unique_output,
                        ),
                        limit=edge.pooling.size,
                    )
                    tasks.append(
                        Attempt.run_many(
                            challenge=parent_attempt.challenge,
                            attempt_config=fix_attempt_config.model_copy(deep=True),
                            raise_exception=False,
                            fixing=attempts_to_use,
                            n_times=fix_attempt_config.attempts,
                        )
                    )

            responses = await tqdm_asyncio.gather(
                *tasks,
                desc=f"[{challenge_id}] Processing fix attempts",
                file=TqdmLogfire(),
            )
            for r in responses:
                local_attempts.extend(r)

            start_eval = time.time()
            took_level = time.time() - start_level
            eval_attempts(
                attempts=local_attempts,
                config=fix_attempt_config,
                plot=PLOT,
                time_took_ms=(took_level * 1000),
            )
            logfire.debug(
                f"[{challenge_id}] eval took {(time.time() - start_eval)} secs"
            )
            all_attempts.extend(local_attempts)
            # now see if you have a solution
            if has_perfect_attempts(all_attempts):
                return all_attempts

            if fix_attempt_config.include_all_attempts_in_fixes:
                parent_attempts = all_attempts
            else:
                parent_attempts = local_attempts

            # now run the fixes
            all_attempts.extend(
                await run_fixes_tree(
                    parent_attempts=parent_attempts,
                    edges=fix_attempt_config.fixes,
                    warm_cache=warm_cache,
                )
            )

            dedup_attempts(all_attempts)
            if has_perfect_attempts(all_attempts):
                return all_attempts

    logfire.debug(f"ALL ATTEMPTS LEN: {len(all_attempts)}")
    return all_attempts


def dedup_attempts(attempts: list[Attempt]) -> list[Attempt]:
    has_seen: set[str] = set()
    _all_attempts = []
    for a in attempts:
        if a.id not in has_seen:
            _all_attempts.append(a)
        has_seen.add(a.id)

    return _all_attempts


async def run_tree(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    warm_cache_root: bool,
    warm_cache_fix: bool,
) -> list[Attempt]:
    # run DFS on this tree

    all_attempts: list[Attempt] = []
    for root_attempt_config in tree:
        start_level = time.time()
        message = f"[{challenge.id}] running root node with {root_attempt_config.attempts} attempts."
        print(message)
        logfire.debug(message)
        local_attempts = await Attempt.run_many(
            challenge=challenge,
            attempt_config=root_attempt_config,
            raise_exception=False,
            fixing=[],
            n_times=root_attempt_config.attempts,
        )
        start_eval = time.time()
        took_level = time.time() - start_level
        eval_attempts(
            attempts=local_attempts,
            config=root_attempt_config,
            plot=PLOT,
            time_took_ms=(took_level * 1000),
        )
        logfire.debug(f"[{challenge.id}] eval took {(time.time() - start_eval)} secs")
        all_attempts.extend(local_attempts)
        all_attempts = dedup_attempts(all_attempts)

        # now see if you have a solution
        if has_perfect_attempts(all_attempts):
            return all_attempts

        # now run the fixes
        if root_attempt_config.include_all_attempts_in_fixes:
            parent_attempts = all_attempts
        else:
            parent_attempts = local_attempts
        all_attempts.extend(
            await run_fixes_tree(
                parent_attempts=parent_attempts,
                edges=root_attempt_config.fixes,
                warm_cache=warm_cache_fix,
            )
        )
        all_attempts = dedup_attempts(all_attempts)

        # now see if you have a solution
        if has_perfect_attempts(all_attempts):
            return all_attempts

    return dedup_attempts(all_attempts)


def get_grids_from_attempt(attempt: Attempt) -> list[GRID]:
    challenge = attempt.challenge
    if len(challenge.test) == 1:
        return [attempt.test_attempt]
    transform_results = run_python_transform_sync(
        code=attempt.python_code_str,
        grid_lists=[deepcopy(test.input) for test in challenge.test],
        timeout=5,
        raise_exception=True,
    )
    logfire.debug(
        f"[{challenge.id}] FINAL: Transform results took {transform_results.latency_ms:.2f} ms"
    )
    return transform_results.transform_results


async def solve_challenge(
    tree: list[RootAttemptConfig], challenge: Challenge, url: str = None
) -> tuple[list[GRID], list[GRID]]:
    if url:
        env_vars = {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "LOGFIRE_TOKEN": os.environ.get("LOGFIRE_TOKEN"),
            "NEON_DB_DSN": os.environ.get("NEON_DB_DSN"),
        }
        if "KAGGLE" in os.environ:
            # RECORD 0 LOGS
            env_vars["KAGGLE"] = os.environ["KAGGLE"]
        try:
            async with httpx.AsyncClient(timeout=3000) as client:
                r = await client.post(
                    url,
                    json={
                        "tree": TypeAdapter(list[RootAttemptConfig]).dump_python(
                            tree, mode="json"
                        ),
                        "challenge": Challenge.model_dump(challenge, mode="json"),
                        "env_vars": env_vars,
                    },
                )
                j = r.json()
            print(f"[{challenge.id}] solved")
            return j
        except Exception as e:
            logfire.debug(f"ERROR RUNNING PYTHON: {e}")
            pass

    run_id = f"run_{random_string(10)}"
    started_at_ms = time.time() * 1000

    attempts = await run_tree(
        tree=tree, challenge=challenge, warm_cache_root=True, warm_cache_fix=False
    )
    attempts = dedup_attempts(attempts)

    # get the number and cost from all of these attempts
    total_cost_cents = sum(a.cost_cents for a in attempts)
    logfire.debug(
        f"[{challenge.id}] DONE: n attempts: {len(attempts)}, total cost cents: {total_cost_cents}"
    )

    ended_at_ms = time.time() * 1000

    if os.environ.get("NEON_DB_DSN"):
        await Attempt.insert_run(
            run_id=run_id, started_at_ms=started_at_ms, ended_at_ms=ended_at_ms
        )
        await Attempt.insert_many(attempts=attempts, run_id=run_id)

    top_two = get_best_attempts(
        attempts=attempts, k_top=2, unique_code=True, unique_output=True
    )

    if len(top_two) == 1:
        top_two.append(top_two[0])

    first_solution = top_two[0]
    second_solution = top_two[1]

    if PLOT:
        first_solution.plot(ignore_fixing=True)
        second_solution.plot(ignore_fixing=True)

    return get_grids_from_attempt(first_solution), get_grids_from_attempt(
        second_solution
    )


async def solve_challenge_server(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    env_vars: dict[str, str],
) -> tuple[list[GRID], list[GRID]]:
    for k, v in env_vars.items():
        os.environ[k] = v
    res = await solve_challenge(tree=tree, challenge=challenge)
    for k in env_vars.keys():
        del os.environ[k]
    return res


class CacheData(BaseModel):
    redis_dsn: str
    run_id: str


class ChallengeStatus(str, Enum):
    queued = "queued"
    running = "running"
    errored = "errored"
    done = "done"


class ChallengeItem(BaseModel):
    status: ChallengeStatus
    queued_at_ms: float
    started_at_ms: float | None
    errored_at_ms: float | None
    done_at_ms: float | None
    solution_attempts: tuple[list[GRID], list[GRID]] | None
    last_ping_at_ms: float | None


async def solve_challenge_background(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    cache_data: CacheData,
    environ_data: dict[str, str],
    url: str = None,
) -> tuple[list[GRID], list[GRID]]:
    if url:
        async with httpx.AsyncClient(timeout=3600) as client:
            r = await client.post(
                url,
                json={
                    "tree": TypeAdapter(list[RootAttemptConfig]).dump_python(
                        tree, mode="json"
                    ),
                    "challenge": Challenge.model_dump(challenge, mode="json"),
                    "cache_data": cache_data.model_dump(mode="json"),
                    "environ_data": environ_data,
                },
            )
            j = r.json()
            debug(j)
            # TODO run retry logic here?
        return j

    for k, v in environ_data.items():
        os.environ[k] = v

    redis_client = redis.Redis.from_url(cache_data.redis_dsn)

    # confirm it hasn't already been done or solved
    key = f"{cache_data.run_id}:{challenge.id}"
    challenge_item = await redis_client.get(key)
    if not challenge_item:
        now = time.time() * 1000
        challenge_item = ChallengeItem(
            status=ChallengeStatus.running,
            queued_at_ms=now,
            started_at_ms=now,
            errored_at_ms=None,
            done_at_ms=None,
            solution_attempts=None,
            last_ping_at_ms=now,
        )
    else:
        challenge_item = ChallengeItem.model_validate_json(challenge_item)
        if challenge_item.status not in [
            ChallengeStatus.queued,
            ChallengeStatus.errored,
        ]:
            raise Exception(f"Invalid challenge status: {challenge_item.status.value}")
        now = time.time() * 1000
        challenge_item = ChallengeItem(
            status=ChallengeStatus.running,
            queued_at_ms=challenge_item.queued_at_ms,
            started_at_ms=now,
            errored_at_ms=challenge_item.errored_at_ms,
            done_at_ms=challenge_item.done_at_ms,
            solution_attempts=challenge_item.solution_attempts,
            last_ping_at_ms=now,
        )

    await redis_client.set(key, challenge_item.model_dump_json())
    try:
        solution_attempts = await solve_challenge(tree=tree, challenge=challenge)
        challenge_item = ChallengeItem.model_validate_json(await redis_client.get(key))
        now = time.time() * 1000
        await redis_client.set(
            key,
            ChallengeItem(
                status=ChallengeStatus.done,
                queued_at_ms=challenge_item.queued_at_ms,
                started_at_ms=challenge_item.started_at_ms,
                errored_at_ms=challenge_item.errored_at_ms,
                done_at_ms=now,
                solution_attempts=solution_attempts,
                last_ping_at_ms=now,
            ).model_dump_json(),
        )
        return solution_attempts
    except Exception as e:
        logfire.debug(f"ERROR CATCHING ATTEMPTS: {e=}, {traceback.format_exc()}")
        now = time.time() * 1000
        await redis_client.set(
            key,
            ChallengeItem(
                status=ChallengeStatus.errored,
                queued_at_ms=challenge_item.queued_at_ms,
                started_at_ms=challenge_item.started_at_ms,
                errored_at_ms=now,
                done_at_ms=challenge_item.done_at_ms,
                solution_attempts=challenge_item.solution_attempts,
                last_ping_at_ms=now,
            ).model_dump_json(),
        )
        raise e
