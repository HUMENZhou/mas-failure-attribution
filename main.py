"""Entry point for running iterative attack/diagnosis attribution workflows.

This module orchestrates:
- task loading from a parquet dataset,
- round-based coding task execution,
- evaluation-driven branching into attack or diagnosis analysis,
- replay from recovery snapshots,
- and final attribution result persistence.
"""

# Standard library imports.
import argparse
import ast
from asyncio import Semaphore
import importlib
import json
import shutil
from pathlib import Path
from typing import Type

# Third-party library imports.
import datasets
from pipeline.multimodal_task.multimodal_eval import load_eval_results_new, run_eval_tasks_new
from sandbox_fusion import set_dataset_endpoint, set_sandbox_endpoint
import asyncio
from tqdm.asyncio import tqdm

# Project-local imports: adapters and monitors.
from adapter.base_adapter import BaseAdapter
from monitor.attack_monitor import AttackMonitor
from monitor.base_monitor import BaseMonitor

# Project-local imports: pipeline stages.
from pipeline.coding.attack import attack_analysis, get_attack_analysis
from pipeline.coding.diagnose import diagnose_analysis, get_diagnose_analysis
from pipeline.coding.eval import load_eval_results, run_eval_tasks
from pipeline.coding.run import run_coding_task

# Project-local imports: utilities.
from utils.common import match_info, read_json_file, save_final_result, write_json_file
from utils.logging import handler
from utils.logging import logger

from utils.prompts import REPLAY_PROMPT
from utils.task_record import normalize_parquet_task_row

def _load_backend(name: str) -> Type[BaseAdapter]:
    """Load a backend adapter class by backend name.

    Args:
        name: Backend identifier that maps to ``adapter.<name>.core``.

    Returns:
        A class object that implements the ``BaseAdapter`` interface.
    """
    backend = importlib.import_module(f"adapter.{name}.core")
    return getattr(backend, f'{name}Adapter')


async def main(args):
    """Run the full multi-round attribution pipeline.

    The function performs round-0 execution, evaluates outcomes, and then
    iteratively applies attack or diagnosis analysis based on previous-round
    evaluation results. It replays tasks from recovery snapshots and persists
    final attribution records when behavior flips between rounds.

    Args:
        args: Parsed CLI arguments used to configure dataset, backend,
            workspace/output directories, and round execution controls.
    """

    # step0: load dataset
    ## TODO: 非.parquet的数据集需要处理
    dataset = datasets.load_dataset(
        "parquet", data_files={"train": args.dataset}, split="train"
    )
    logger.info(f"Loaded {len(dataset)} tasks from {args.dataset}")
    tasks = dataset.to_list()
    tasks = [
        normalize_parquet_task_row(t, dataset_path=args.dataset)
        for t in tasks
    ]
        
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
        logger.info(f"Using {len(tasks)} tasks (max_samples={args.max_samples})")
    Backend = _load_backend(args.backend)
    backend = Backend()

    data_source = tasks[0]["data_source"]
    workspace_root: Path = args.workspace
    output_root: Path = args.output

    skip_existing = args.skip_existing
    max_rounds = args.max_rounds
    rollout_only = args.rollout
    run_attack = args.run_mode in ("all", "attack")
    run_diagnose = args.run_mode in ("all", "diagnose")
    logger.info(
        f"run_mode={args.run_mode}: attack={'on' if run_attack else 'off'}, "
        f"diagnose={'on' if run_diagnose else 'off'}"
    )

    use_concurrency = args.concurrent is not None and args.concurrent > 1
    concurrency = args.concurrent if use_concurrency else 1

    # covert to absolute path 
    if not workspace_root.is_absolute():
        workspace_root = workspace_root.absolute()
    if not output_root.is_absolute():
        output_root = output_root.absolute()

    # step1: run dataset tasks in multi-agent system, get running trajectory and eval results
    # TODO：重构代码，这段主要是为了获取多智能体系统运行轨迹，main函数太长了
    coros = []
    semaphore = Semaphore(concurrency)
    for task in tasks:
        task_id = task["task_id"].replace("/", "_")
        workspace = workspace_root / data_source / f"round_0" / task_id
        output = output_root / data_source / f"round_0" / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)

        logger.info('No recovery info, initializing new monitor...')
        recovery_path = output / 'recovery'
        monitor = BaseMonitor(recovery_path, workspace, backend)
        # TODO: 这里需要根据不同的数据集选择不同的run_coding_task函数
        coros.append(run_coding_task(
            task,
            workspace,
            output,
            backend,
            skip_existing=skip_existing,
            monitor=monitor,
            semaphore=semaphore
        ))

    eval_path = output_root / data_source / "round_0"
    await tqdm.gather(*coros)

    # TODO：这里看如何根据数据集 选择对应的结果评估函数
    if args.backend == "MagenticOne":
        # TODO：更改函数名称，run_eval_tasks_new
        run_eval_tasks_new(eval_path, data_source=data_source, skip_existing=skip_existing)
        eval_results = load_eval_results_new(eval_path, data_source)
    else:
    # Start to eval round 0
        await run_eval_tasks(eval_path, data_source=data_source, semaphore=semaphore, skip_existing=skip_existing)
        eval_results, msg_results = load_eval_results(eval_path, data_source)
    
    if rollout_only:
        return


    
    # step2 multi-point injection and playback verification TODO：重构代码，现在太长了，抽象出单独的函数
    # ROUND i >= 1: divide eval results of round 0 into success / fail
    # sucess -> attack pipeline
    # fail -> diagnose pipeline
    # current implementation is for testing replay function
    completed_tasks = []

    for current in range(1, max_rounds + 1):
        length = len(tasks)
        batch_size = concurrency * 100
        for i in range(0, length+batch_size, batch_size):
            coros = []
            for task in tasks[i:i+batch_size]:       

                async def _run(task):
                    task_id = task["task_id"].replace("/", "_")
                    if task_id in completed_tasks:
                        return
                    
                    logger.info(f'Round {current}: Start to process Task {task_id}')
                    last_round_output = output_root / data_source / f"round_{current-1}" / task_id
                    if not last_round_output.exists():
                        raise FileNotFoundError(f'last round output not exists for {task_id}')

                    last_round_log = read_json_file(last_round_output / 'log.json') 
                    last_round_log = str(last_round_log).replace(f'round_{current-1}', f'round_{current}')
                    last_round_log = ast.literal_eval(last_round_log)
                    output = output_root / data_source / f"round_{current}" / task_id
                    output.mkdir(parents=True, exist_ok=True)
                    log = output / 'log.json'
                    attack_path = output / 'attack_analysis.json'
                    diagnose_path = output / 'diagnose_analysis.json'
                    if log.exists():
                        if skip_existing:
                            logger.info(f'Log for task {task_id} exists, skipping this round...')
                            return
                    
                    # step 2 branch 1: sucess -> attack pipeline
                    if eval_results[task_id]:
                        if not run_attack:
                            logger.info(
                                f'run_mode={args.run_mode}: skip attack for success task {task_id}'
                            )
                            shutil.copytree(last_round_output, output, dirs_exist_ok=True,)
                            return

                        logger.info(f'Last round processed as success for {task_id}, start the attack process...')
                        
                        workspace = workspace_root / data_source / f"round_{current}" / f'{task_id}_attack_analysis'
                        workspace.mkdir(parents=True, exist_ok=True)
                        previous_injections_path = last_round_output / 'attack_analysis.json'
                        previous_injections = (
                            read_json_file(previous_injections_path) 
                                if previous_injections_path.exists() else []
                        )

                        is_success = await attack_analysis(
                            task=last_round_log,
                            workspace=workspace,
                            output=output,
                            backend=backend,
                            skipping_exists=skip_existing,
                            injection_history=previous_injections,
                            message=msg_results[task_id],
                            semaphore=semaphore
                        )

                        if not is_success:
                            logger.info(f'Attack Analysis Failed, skipping this round...')
                            shutil.copytree(
                                last_round_output,
                                output,
                                dirs_exist_ok=True
                            )
                            return

                        replay_info = get_attack_analysis(output)

                    # step 2 branch 2: fail -> diagnose pipeline
                    else:
                        if not run_diagnose:
                            logger.info(
                                f'run_mode={args.run_mode}: skip diagnose for failed task {task_id}'
                            )
                            shutil.copytree(
                                last_round_output,
                                output,
                                dirs_exist_ok=True,
                            )
                            return

                        logger.info(f'Last round processed as failure for {task_id}, start the diagnosis process...')
                        
                        workspace = workspace_root / data_source / f"round_{current}" / f'{task_id}_diagnose_analysis'
                        workspace.mkdir(parents=True, exist_ok=True)

                        previous_injections_path = last_round_output / 'diagnose_analysis.json'
                        previous_injections = (
                            read_json_file(previous_injections_path) 
                                if previous_injections_path.exists() else []
                        )

                        is_success = await diagnose_analysis(
                            task=last_round_log,
                            workspace=workspace,
                            output=output,
                            backend=backend,
                            skipping_exists=skip_existing,
                            injection_history=previous_injections,
                            message=msg_results[task_id],
                            semaphore=semaphore,
                            diagnose_mode=args.diagnose_mode,
                        )
                        if not is_success:
                            logger.info(f'Diagnose Analysis Failed, skipping this round...')
                            shutil.copytree(
                                last_round_output,
                                output,
                                dirs_exist_ok=True
                            )
                            return
                        replay_info = get_diagnose_analysis(output)
                    
                    monitor = AttackMonitor(recovery_path, workspace, backend, replay_info[-1], last_round_log)
                    result = await run_coding_task(
                        task,
                        workspace,
                        output,
                        backend,
                        skip_existing=skip_existing,
                        monitor=monitor,
                        semaphore=semaphore
                    )
                    if not result:
                        logger.info(f'Replay Failed, skipping this round...')
                        shutil.copytree(
                            last_round_output,
                            output,
                            dirs_exist_ok=True
                        )
                        return
                
                coros.append(_run(task))
        
            await tqdm.gather(*coros)

        # save last round's eval results
        last_eval_results = eval_results
        eval_path = output_root / data_source / f"round_{current}"
        if args.backend == "MagenticOne":
            run_eval_tasks_new(eval_path, data_source=data_source, skip_existing=skip_existing)
            eval_results = load_eval_results_new(eval_path, data_source)
        else:
            await run_eval_tasks(eval_path, data_source=data_source, semaphore=semaphore, skip_existing=skip_existing)
            eval_results, msg_results = load_eval_results(eval_path, data_source)
        
        for task_id in eval_results:
            if task_id not in last_eval_results:
                continue
            # step 2 branch 3: behavior flip -> final attribution
            if eval_results[task_id] ^ last_eval_results[task_id]:
                output = output_root / data_source / f"round_{current}" / task_id
                if last_eval_results[task_id]: # from success to fail, 
                    if not run_attack:
                        logger.info(
                            f'run_mode={args.run_mode}: skip attack finalization for {task_id}'
                        )
                        continue
                    logger.info(f'[Round {current}] Attack result eval changed to failure, diagnosing...')
                    last_round_output = output_root / data_source / f"round_{current}" / task_id
                    if not last_round_output.exists():
                        raise FileNotFoundError(f'last round output not exists for {task_id}')
                    last_round_log = read_json_file(last_round_output / 'log.json') 
                    """
                    workspace = workspace_root / data_source / f"round_{current}" / f'{task_id}_diagnose_analysis'
                    workspace.mkdir(parents=True, exist_ok=True)
                    is_success = await diagnose_analysis(
                        task=last_round_log,
                        workspace=workspace,
                        output=output,
                        backend=backend,
                        skipping_exists=skip_existing,
                        semaphore=semaphore
                    )"""
                    try:
                        final_info = get_attack_analysis(output)
                    except Exception as e:
                        logger.error(f"Error occurred while analyzing attack results for {task_id}: {e}")
                        continue
                    """
                    if is_success:
                        diagnose_info = get_diagnose_analysis(output)
                        if match_info(final_info, diagnose_info):
                           logger.info(f'Direct diagnose success, regarding as easy injection...')
                           continue """
                else:                               # from fail to success
                    if not run_diagnose:
                        logger.info(
                            f'run_mode={args.run_mode}: skip diagnose finalization for {task_id}'
                        )
                        continue
                    last_round_output = output_root / data_source / f"round_{current-1}" / task_id
                    if not last_round_output.exists():
                        raise FileNotFoundError(f'last round output not exists for {task_id}')
                    last_round_log = read_json_file(last_round_output / 'log.json') 
                    try:
                        final_info = get_diagnose_analysis(output)
                    except Exception as e:
                        logger.error(f"Error occurred while analyzing diagnose results for {task_id}: {e}")
                        continue

                completed_tasks.append(task_id)
                save_final_result(output_root / 'final_results', last_round_log, final_info)
            else:
                logger.info(f'Eval Result remains the same, Attack/Diagnose Fail...')
        
if __name__ == "__main__":
    handler.doRollover()
    set_sandbox_endpoint("http://localhost:8080/")
    set_dataset_endpoint("http://localhost:8080/online_judge/")
    parser = argparse.ArgumentParser(description="Universal attack and diagnosis framework")
    # 设置数据集
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset path",
    )
    # 设置多智能体系统
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="MAS backend name",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="Working directory path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory path")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of rounds")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples")
    
    # TODO: concurrency
    parser.add_argument("--concurrent", type=int, default=None, help="Enable concurrent processing")
    parser.add_argument("--skip_existing", "-s", action="store_true")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["all", "attack", "diagnose"],
        default="all",
        help=(
            "Pipeline branch control: all (success->attack, fail->diagnose), "
            "attack (only attack on success), diagnose (only diagnose on failure)"
        ),
    )
    parser.add_argument(
        "--diagnose-mode",
        type=str,
        choices=["default", "critic"],
        default="default",
        help="Diagnose analysis prompt mode: default (direct) or critic (CRITIC tool-interactive)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))