"""Extended evaluation utilities supporting both code execution and LLM-based semantic comparison."""

import json
from pathlib import Path
from typing import Dict, Any

from utils.common import read_json_file
from utils.logging import logger
from sandbox_fusion import (
    RunCodeRequest,
    RunStatus,
    run_code,
)
from adapter.MagenticOne.core import _build_model_client


def code_exec(code: str, test: str):
    """Execute candidate solution with provided tests in sandbox runtime."""
    import base64
    base64_content = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    request = RunCodeRequest(
        language="pytest",
        code=test,
        files={"solution.py": base64_content},
    )
    try:
        return run_code(request, client_timeout=60)
    except Exception as e:
        logger.error(f"Error in code_exec: {e}", exc_info=True)
        return None


def llm_semantic_eval(prediction: str, ground_truth: str, question: str) -> bool:
    """
    Use an LLM to compare prediction and ground truth semantically.
    Returns True if the prediction is considered correct based on the ground truth.
    """
    if not prediction or not ground_truth:
        return False

    # Cheap exact match first to save tokens.
    if prediction.strip().lower() == ground_truth.strip().lower():
        return True

    system_prompt = """You are a precise evaluator. Your task is to determine if the 'Model Prediction' correctly answers the 'Question' by comparing it with the 'Ground Truth'.

    Rules:
    1. The prediction does not need to be word-for-word identical to the ground truth.
    2. If the prediction conveys the same meaning, facts, or numerical value as the ground truth, mark it as correct.
    3. If the prediction is irrelevant, factually wrong, or contradicts the ground truth, mark it as incorrect.
    4. Return ONLY a JSON object with a single key "is_correct" (boolean). Do not add any explanation.

    Example Output: {"is_correct": true}
    """

    user_prompt = f"""
    Question: {question}
    Ground Truth: {ground_truth}
    Model Prediction: {prediction}
    """

    try:
        import asyncio

        from autogen_core.models import SystemMessage, UserMessage

        async def _call() -> str:
            client = _build_model_client()
            result = await client.create(
                [
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt, source="user"),
                ],
            )
            raw = result.content
            if not isinstance(raw, str):
                raise TypeError(f"Expected string completion, got {type(raw)}")
            return raw

        content = asyncio.run(_call())
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        if "{" in text and "}" in text:
            start, end = text.find("{"), text.rfind("}")
            if end > start:
                text = text[start : end + 1]
        result_json = json.loads(text)
        return bool(result_json.get("is_correct", False))

    except Exception as e:
        logger.warning(f"LLM eval failed for question: {e}. Falling back to strict match.")
        return False


def run_universal_eval_task(task: dict) -> tuple[str, bool]:
    """
    Evaluate a task based on its type.
    - If 'test' field exists and is not empty -> Code Execution Eval
    - Otherwise -> LLM Semantic Eval
    """
    task_id = task["question_ID"]
    prediction = task.get("model_prediction", "")
    ground_truth = task.get("ground_truth", "")
    question = task.get("question", "")
    test_code = task.get("test", "")

    # Coding task when non-empty pytest body is present.
    if test_code and test_code.strip():
        result = code_exec(prediction, test_code)
        if result:
            return task_id, result.status == RunStatus.Success
        else:
            return task_id, False
    else:
        # Non-code tasks: LLM semantic judge.
        is_correct = llm_semantic_eval(prediction, ground_truth, question)
        return task_id, is_correct


def run_eval_tasks_new(
        eval_path: Path,
        data_source: str,
        skip_existing: bool = True,
) -> Dict[str, bool]:
    """
    Evaluate all tasks using the universal evaluator (Code + LLM).
    """
    logger.info(f'Starting universal eval for {data_source} in {eval_path}...')
    filename = f"eval_{data_source}.json"
    save_path = eval_path / filename

    if save_path.exists():
        if skip_existing:
            logger.info(f'Eval result {filename} exists, skipping...')
            return {}
        else:
            logger.info(f'Eval result {filename} exists, overriding...')

    eval_log_paths = [p / 'log.json' for p in eval_path.iterdir() if p.is_dir()]
    tasks = [read_json_file(p) for p in eval_log_paths if p.exists()]

    results = {}
    for task in tasks:
        task_id, is_correct = run_universal_eval_task(task)
        results[task_id] = is_correct
        logger.info(f"Task {task_id}: {'PASS' if is_correct else 'FAIL'}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f'Universal eval finished. Results saved to {save_path}')
    return results


def load_eval_results_new(eval_path: Path, data_source: str):
    """Load the new evaluation results."""
    return read_json_file(eval_path / f"eval_{data_source}.json")
