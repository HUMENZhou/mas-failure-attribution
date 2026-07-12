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
from sandbox_fusion import (
    RunCodeRequest,
    RunStatus,
    run_code,
    run_code_async,
)

def code_exec(code: str, test: str):
    """Execute candidate solution with provided tests in sandbox runtime."""

    base64_content = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    request = RunCodeRequest(
        compile_timeout=60,
        run_timeout=60,
        language="pytest",
        code=test,
        files={"solution.py": base64_content},
    )
    try:
        return run_code(request, client_timeout=60)
    except Exception as e:
        # Log exception details
        logger.error(f"Error in code_exec_async: {e}", exc_info=True)
        # Depending on needs, you can return a custom error object or None
        print(f"Error in code_exec_async: {e}")
        return None


async def code_exec_async(code: str, test: str):
    base64_content = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    request = RunCodeRequest(
        compile_timeout=60,
        run_timeout=60,
        language="pytest",
        code=test,
        files={"solution.py": base64_content},
    )
    try:
        return await run_code_async(request, client_timeout=60)
    except Exception as e:
        # Log exception details
        logger.error(f"Error in code_exec_async: {e}", exc_info=True)
        # Depending on needs, you can return a custom error object or None
        print(f"Error in code_exec_async: {e}")
        return None

async def run_correctness_eval_task(
    task: dict, semaphore: asyncio.Semaphore = None
) -> tuple[str, bool]:
    """Run one task evaluation and return ``(task_id, passed)``."""
    async with semaphore:
        solution = task["model_prediction"]
        test = task["test"]
        result = await code_exec_async(solution, test)
        if result is None:
            logger.error(f"Code execution failed for task {task['question_ID']}")
            return task["question_ID"], False, "Code execution error"
        logger.info(f"Eval result for task {task['question_ID']}: {result.status == RunStatus.Success}, message: {str(result)}")
        return task["question_ID"], result.status == RunStatus.Success, result.run_result.stdout


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
    