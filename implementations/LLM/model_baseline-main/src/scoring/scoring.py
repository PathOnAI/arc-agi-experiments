import argparse
from typing import List, Tuple
from pathlib import Path
from typing import List, Tuple, Dict
import json

class ARCScorer:
    def __init__(self, task_dir: str, submission_dir: str, print_logs: bool = False, results_dir: str = None):
        self.task_dir = Path(task_dir)
        self.submission_dir = Path(submission_dir)
        self.print_logs = print_logs
        self.solutions = self._load_solutions()
        self.results_dir = Path(results_dir) if results_dir else None

    def print_log(self, message: str) -> None:
        if self.print_logs:
            print(message)

    def _load_solutions(self) -> Dict[str, List[List[List[int]]]]:
        solutions = {}
        for solution_file in self.task_dir.glob('*.json'):
            task_id = solution_file.stem
            with solution_file.open() as f:
                solutions[task_id] = json.load(f)
        return solutions

    def score_task(self, task_id: str, submission_path: Path) -> float:
        """
        Scores a single task submission against the solutions.
        """
        with submission_path.open() as f:
            task_submission = json.load(f)

        task_score = 0
        num_pairs = len(task_submission)

        for pair_index, pair_attempts in enumerate(task_submission):
            pair_correct = False
            for attempt in pair_attempts:

                if pair_attempts[attempt] == []:
                    self.print_log(f"    No prediction for {task_id}, pair {pair_index}, attempt {attempt}")
                    continue

                if pair_attempts[attempt] == self.solutions[task_id]['test'][pair_index]['output']:
                    pair_correct = True
                    break

            if pair_correct:
                task_score += 1

        return task_score / num_pairs

    def score_submission(self) -> Tuple[float, int]:
        """
        Read a submission from file, score it, then return the score
        """
        self.print_log(f"Scoring {self.submission_dir}\n")

        total_score = 0
        total_tasks = 0
        task_results = {}

        for submission_file in self.submission_dir.glob('*.json'):
            task_id = submission_file.stem
            task_score = self.score_task(task_id, submission_file)
            self.print_log(f"Task Id {task_id} score {task_score}")
            total_score += task_score
            total_tasks += 1
            task_results[task_id] = task_score

        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            results_file = self.results_dir / "results.json"
            self.print_log(f"Saving results to {results_file}")
            with results_file.open("w") as f:
                json.dump({
                    "score": total_score,
                    "total_tasks": total_tasks,
                    "task_results": task_results
                }, f)

        return total_score, total_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC Tester")

    parser.add_argument("--task_dir", type=str, help="Task directory which contains the full tasks (including solutions)")
    parser.add_argument("--submission_dir", type=str, help="Submission directory which contains the submissions to score")
    parser.add_argument("--print_logs", action="store_true", help="Printing logs to console (default: False)")
    parser.add_argument("--results_dir", type=str, help="Results directory to save the results (default: None)")

    args = parser.parse_args()

    arc_scorer = ARCScorer(
        task_dir=args.task_dir,
        submission_dir=args.submission_dir,
        print_logs=args.print_logs,
        results_dir=args.results_dir
    )

    arc_scorer.score_submission()