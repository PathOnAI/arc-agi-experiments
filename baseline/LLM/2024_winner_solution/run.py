import asyncio
import json
import os
import time
import typing as T
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from src import logfire
from src.data import build_challenges
from src.logic import (
    CacheData,
    random_string,
    solve_challenge,
    solve_challenge_background,
)
from src.models import GRID, Challenge, RootAttemptConfig
from src.trees import deepseek, experiments, o3


class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID


async def solve_and_write(
    solutions_d: dict[str, list[ChallengeSolution]],
    challenge: Challenge,
    tree: list[RootAttemptConfig],
    solutions_dir: Path,
    solutions_path: str,
) -> None:
    start = time.time()
    print(f"[{challenge.id}] starting challenge...")

    if server_url := os.environ.get("SERVER_URL"):
        url = f"{server_url}/solve_challenge"
    else:
        url = None

    first_solutions, second_solutions = await solve_challenge(
        challenge=challenge,
        tree=tree,
        # url=url,
    )
    solutions_d[challenge.id] = []
    for i in range(len(first_solutions)):
        solutions_d[challenge.id].append(
            ChallengeSolution(
                attempt_1=first_solutions[i],
                attempt_2=second_solutions[i],
            )
        )
    # just write after each challenge in case program crashes
    logfire.debug(
        f"[{challenge.id}] solution",
        challenge_id=challenge.id,
        challenge=challenge,
        solution_d=solutions_d[challenge.id],
    )
    open(solutions_dir / f"{challenge.id}.json", "w").write(
        TypeAdapter(list[ChallengeSolution])
        .dump_json(solutions_d[challenge.id])
        .decode("utf-8")
    )
    # just write to be safe
    open(solutions_path, "w").write(
        TypeAdapter(dict[str, list[ChallengeSolution]])
        .dump_json(solutions_d)
        .decode("utf-8")
    )
    took_secs = time.time() - start
    logfire.debug(f"[{challenge.id}] took {took_secs:.2f} secs to solve and write")


async def process_challenges_with_limit(
    challenges: list[Challenge],
    solutions_d: dict[str, list[ChallengeSolution]],
    tree: list[RootAttemptConfig],
    solutions_dir: Path,
    solutions_path: str,
    max_concurrent: int,
) -> list[T.Any]:
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_solve_and_write(challenge: Challenge) -> None:
        """
        Wrapper for solve_and_write that respects the semaphore limit
        """
        async with semaphore:
            try:
                return await solve_and_write(
                    solutions_d=solutions_d,
                    challenge=challenge,
                    tree=tree,
                    solutions_dir=solutions_dir,
                    solutions_path=solutions_path,
                )
            except Exception as e:
                logfire.debug(f"Error processing challenge: {e}")
                raise

    # Create tasks for all challenges
    tasks = [bounded_solve_and_write(challenge) for challenge in challenges]

    # Run all tasks and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions in results
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        message = f"Encountered {len(errors)} errors during processing"
        logfire.debug(message)
        print(message)

    return results


async def run_from_json(
    *,
    challenges_path: str,
    solutions_path: str,
    temp_solutions_dir_path: str,
    tree: list[RootAttemptConfig],
    limit: int | None,
    offset: int | None = None,
    only_run_ids: set[str] = None,
    max_concurrent: int,
    truth_solutions_path: str | None = None,
) -> None:
    start = time.time()
    challenges = build_challenges(
        challenges_path=Path(challenges_path),
        solutions_path=Path(truth_solutions_path) if truth_solutions_path else None,
    )
    if only_run_ids:
        challenges = {k: challenges[k] for k in only_run_ids}
    if offset:
        challenges = {k: challenges[k] for k in list(challenges)[offset:]}
    if limit:
        # only include the first 10 challenges
        challenges = {k: challenges[k] for k in list(challenges)[:limit]}

    solutions_d: dict[str, list[ChallengeSolution]] = {}
    # run all challenges in parallel to start

    solutions_dir = Path(temp_solutions_dir_path)
    solutions_dir.mkdir(exist_ok=True)

    await process_challenges_with_limit(
        challenges=list(challenges.values()),
        solutions_d=solutions_d,
        tree=tree,
        solutions_dir=solutions_dir,
        solutions_path=solutions_path,
        max_concurrent=max_concurrent,
    )

    # iterate through solutions dir and load in the solutions? or just use solutions_d
    open(solutions_path, "w").write(
        TypeAdapter(dict[str, list[ChallengeSolution]])
        .dump_json(solutions_d)
        .decode("utf-8")
    )
    message = f"FINAL: took {(time.time() - start):.2f} secs to run {len(challenges)} challenges"
    logfire.debug(message)
    print(message)


async def run() -> None:
    challenges_path = "arc-prize-2024/arc-agi_training_challenges.json"
    truth_solutions_path = "arc-prize-2024/arc-agi_training_solutions.json"
    attempts_solutions_path = "test_data/training_solutions.json"
    await run_from_json(
        challenges_path=challenges_path,
        solutions_path=attempts_solutions_path,
        truth_solutions_path=truth_solutions_path,
        temp_solutions_dir_path="test_data/tmp_solutions",
        # tree=experiments.sonnet_writeup_deep,
        # tree=experiments.sonnet_writeup_shallow,
        # tree=experiments.sonnet_writeup_med,
        # tree=experiments.sonnet_pooling_example,
        # tree=experiments.deepseek_pooling_example,
        # tree=deepseek.prod_kaggle_tree,
        # tree=deepseek.small_baseten_tree,
        tree=o3.small_tree,
        # limit=10,
        # offset=50,
        limit=1,
        offset=0,
        max_concurrent=20,
        only_run_ids={"045e512c"},
    )
    evaluate_solutions(
        attempts_solutions_path=attempts_solutions_path,
        truth_solutions_path=truth_solutions_path,
    )


def evaluate_solutions(attempts_solutions_path: str, truth_solutions_path: str) -> None:
    truth: dict[str, list[GRID]] = json.loads(open(truth_solutions_path).read())
    attempts: dict[str, list[ChallengeSolution]] = TypeAdapter(
        dict[str, list[ChallengeSolution]]
    ).validate_json(open(attempts_solutions_path).read())
    total_count = 0
    correct_count = 0
    for challenge_id, attempt_list in attempts.items():
        truth_grids: list[GRID] = truth[challenge_id]
        for i, truth_grid in enumerate(truth_grids):
            total_count = total_count + 1
            attempt_grids = attempt_list[i]
            if attempt_grids.attempt_1 == truth_grid:
                correct_count = correct_count + 1
            elif attempt_grids.attempt_2 == truth_grid:
                correct_count = correct_count + 1

    print("total count", total_count, "correct count", correct_count)


async def main() -> None:
    await run()


if __name__ == "__main__":
    asyncio.run(main())
