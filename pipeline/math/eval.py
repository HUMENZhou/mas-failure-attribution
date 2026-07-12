"""Task evaluation utilities based on sandboxed correctness execution."""

import asyncio
import base64
import json
import os
from pathlib import Path
import shutil

from tqdm.asyncio import tqdm
from utils.common import read_json_file
from utils.logging import logger
import re

def math_eval(solution: str, ground_truth: str):
    try:
        ground_truth = re.search(r'\\boxed\{([^}]+)\}', ground_truth)
        ground_truth = ground_truth.group(1).strip() if ground_truth else ""
        answer_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        answer = answer_match.group(1).strip() if answer_match else ""
        return answer == ground_truth
    except Exception as e:
        # Log exception details
        logger.error(f"Error in code_exec_async: {e}", exc_info=True)
        # Depending on needs, you can return a custom error object or None
        print(f"Error in code_exec_async: {e}")
        return False

async def run_correctness_eval_task(
    task: dict, semaphore: asyncio.Semaphore = None
) -> tuple[str, bool]:
    """Run one task evaluation and return ``(task_id, passed)``."""
    async with semaphore:
        solution = task["model_prediction"]
        ground_truth = task["ground_truth"]
        result = math_eval(solution, ground_truth)
        logger.info(f"Eval result for task {task['question_ID']}: {result}")
        return task["question_ID"], result, ""

async def run_eval_tasks(
    eval_path: Path,
    data_source: str,
    semaphore: asyncio.Semaphore = None,
    skip_existing: bool = True,
) -> dict[str, bool]:
    """
    Evaluate all task logs under one round directory and save pass/fail map.

    Args:
        eval_path: Directory containing per-task subdirectories with ``log.json``.
        data_source: Dataset source name used in evaluation output filename.
        skip_existing: Whether to skip when evaluation result already exists.

    Returns:
        Mapping from task id to pass/fail status, or ``None`` when skipped.
    """
    logger.info(f'Starting to eval tasks in {eval_path}...')
    filename = f"eval_{data_source}.json"
    save_path = eval_path / filename
    msg_path = eval_path / f'eval_msg_{data_source}.json'
    if save_path.exists():
        if skip_existing:
            logger.info(f'Eval result for {data_source} exists, skipping...')
            return
        else:
            logger.info(f'Eval result for {data_source} exists, overriding...')

    eval_log_paths = [p / 'log.json' for p in eval_path.iterdir() if p.is_dir()]
    tasks = [read_json_file(p) for p in eval_log_paths if p.exists()]
    coros = [run_correctness_eval_task(task, semaphore) for task in tasks]
    results = await tqdm.gather(*coros)
    status = {task_id: result for task_id, result,_ in results}
    messages = {task_id: message for task_id, _, message in results}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(status, f)
    with open(msg_path, "w") as f:
        json.dump(messages, f)
    logger.info(f'Ending to eval tasks in {eval_path}, results saved in {save_path}...')

def load_eval_results(
    eval_path: Path,
    data_source: str,
):
    """Load previously saved evaluation result mapping for one round."""
    eval_results = read_json_file(eval_path / f"eval_{data_source}.json")
    msg_results = read_json_file(eval_path / f"eval_msg_{data_source}.json")
    return eval_results, msg_results
    