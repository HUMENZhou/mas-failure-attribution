"""Shared MagenticOne / AutoGen runtime helpers (from te_mg_2_parquet, pipeline-safe)."""

from __future__ import annotations

import asyncio
import configparser
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

# Disable third-party experiment trackers before heavy imports.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WEAVE_DISABLE_AUTO_LOGGING", "true")

warnings.filterwarnings(
    "ignore",
    message='.*Field "model_.*" has conflict with protected namespace.*',
)

WINDOWS_ORCHESTRATOR_ENV_APPEND = """
ENVIRONMENT CONSTRAINT (Windows — non-negotiable for this run):
- ComputerTerminal runs on Windows. Do NOT direct Coder or the team to use ```sh / ```bash, nor POSIX one-liners (curl/wget/grep/sed).
- To fetch or read public web pages / follow links, prefer WebSurfer.
- For local files with known absolute paths, prefer FileSurfer or Python.
- When terminal scripting is needed, instruct Coder to use ```powershell only, or ```python blocks; never assume bash/sh is available.
"""


def patch_magentic_one_orchestrator_prompts_for_windows() -> None:
    try:
        import autogen_agentchat.teams._group_chat._magentic_one._prompts as m1p
    except ImportError:
        return
    for name in (
        "ORCHESTRATOR_PROGRESS_LEDGER_PROMPT",
        "ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT",
        "ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT",
    ):
        cur = getattr(m1p, name, "")
        if isinstance(cur, str) and cur:
            setattr(m1p, name, cur + WINDOWS_ORCHESTRATOR_ENV_APPEND)


patch_magentic_one_orchestrator_prompts_for_windows()

WINDOWS_TEAM_TASK_PREFIX = (
    "[Team environment: Windows. Prefer WebSurfer for web pages and links; "
    "prefer FileSurfer or Python for local files. Do not assign bash/sh/curl POSIX shell to the code executor.]\n\n"
)

WINDOWS_CODER_CODE_EXECUTOR_INSTRUCTION = """
CRITICAL — ComputerTerminal uses Microsoft Windows (LocalCommandLineCodeExecutor). Code you emit is executed as-is on Windows.

RULES (follow in order):
1) Web: Prefer asking WebSurfer to open URLs / read HTML / follow links. Do not default to terminal HTTP.
2) Local files: Prefer FileSurfer or Python with absolute paths given in the task.
3) Shell blocks for the terminal (ONLY when shell is truly needed):
   - Use EXCLUSIVELY fenced blocks labeled ```powershell ... ``` — NEVER ```sh, ```bash, ```zsh, or bare POSIX.
   - Do NOT use `curl ...` or `wget ...` in POSIX form. If you must call curl from PowerShell, use `curl.exe` with Windows rules, or prefer:
     `Invoke-WebRequest -UseBasicParsing -Uri 'https://example.com' | Select-Object -ExpandProperty Content`
4) Portable option: Prefer ```python ... ``` with stdlib `urllib.request.urlopen` or `requests` when available.
5) Never assume `bash`, `sh`, `/bin/*`, or Unix-only paths exist.

BAD (do not output):
```sh
curl -s https://example.com
```

GOOD (PowerShell):
```powershell
(Invoke-WebRequest -UseBasicParsing -Uri 'https://example.com').Content
```

GOOD (Python):
```python
import urllib.request
print(urllib.request.urlopen("https://example.com").read().decode())
```
"""


def gaia_attachments_root(repo_root: Path | None = None) -> Path:
    raw = os.environ.get("GAIA_ATTACHMENTS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if repo_root is not None:
        return (repo_root / "dataset" / "gaia_dataset_raw").resolve()
    return Path(".").resolve()


def build_gaia_agent_task(sample: Dict[str, Any], attachments_root: Path) -> str:
    question = (sample.get("question") or "").strip()
    fn = (sample.get("file_name") or "").strip()
    fp_rel = (sample.get("file_path") or "").strip()
    if not fn or not fp_rel:
        return question
    abs_path = (attachments_root / fp_rel).resolve()
    exists = abs_path.is_file()
    return (
        f"{question}\n\n"
        "[Attached file for this task (from GAIA metadata)]\n"
        f"- file_name: {fn}\n"
        f"- file_path (relative to GAIA attachments root): {fp_rel}\n"
        f"- resolved absolute path: {abs_path}\n"
        f"- file exists on disk: {exists}\n"
        "Use FileSurfer or local tools with the absolute path when the file exists. "
        "If file exists is false, the attachment was not downloaded locally—report that clearly.\n"
    )


def gaia_preamble_for_pipeline_task(task: Dict[str, Any], attachments_root: Path) -> str:
    """Append-only GAIA attachment block; caller prepends to full `idea` to avoid duplicating question."""
    block = build_gaia_agent_task(
        {
            "question": "",
            "file_name": task.get("file_name"),
            "file_path": task.get("file_path"),
        },
        attachments_root,
    ).strip()
    if not block:
        return ""
    return block + "\n\n"


def make_local_command_line_code_executor(
    workspace: Path,
    *,
    timeout: int = 120,
) -> Any:
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    virtual_env_context: SimpleNamespace | None = None
    if sys.platform == "win32":
        current_python = Path(sys.executable).resolve()
        if current_python.is_file() and current_python.name.lower() in ("python.exe", "pythonw.exe"):
            rp = current_python
            scripts = rp.parent / "Scripts"
            bin_path = str(scripts if scripts.is_dir() else rp.parent)
            virtual_env_context = SimpleNamespace(env_exe=str(rp), bin_path=bin_path)
        else:
            conda = os.environ.get("CONDA_PREFIX")
            if conda:
                py = Path(conda) / "python.exe"
                if py.is_file():
                    rp = py.resolve()
                    scripts = rp.parent / "Scripts"
                    bin_path = str(scripts if scripts.is_dir() else rp.parent)
                    virtual_env_context = SimpleNamespace(env_exe=str(rp), bin_path=bin_path)
    kwargs: dict[str, Any] = {"work_dir": str(workspace), "timeout": timeout}
    if virtual_env_context is not None:
        kwargs["virtual_env_context"] = virtual_env_context
    return LocalCommandLineCodeExecutor(**kwargs)


def _text_from_message(msg: Any) -> str:
    from autogen_agentchat.messages import MultiModalMessage, TextMessage

    if isinstance(msg, TextMessage):
        c = msg.content
        return c.strip() if isinstance(c, str) else ""
    if isinstance(msg, MultiModalMessage):
        parts: List[str] = []
        for p in msg.content or []:
            if isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()
    return ""


def extract_model_prediction_from_task_result(task_result: Any) -> str:
    from autogen_agentchat.messages import TextMessage

    messages: Sequence[Any] = getattr(task_result, "messages", None) or ()

    def is_orchestrator(src: Any) -> bool:
        s = (src or "").lower()
        return "orchestrator" in s or s == "magenticoneorchestrator"

    for msg in reversed(messages):
        if not isinstance(msg, TextMessage):
            continue
        if not is_orchestrator(getattr(msg, "source", None)):
            continue
        text = _text_from_message(msg)
        if text:
            return text

    for msg in reversed(messages):
        src = getattr(msg, "source", None)
        if src == "user":
            continue
        text = _text_from_message(msg)
        if text:
            return text
    return ""


def _exception_causes(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = [exc]
    cur: BaseException | None = exc.__cause__
    while cur is not None and len(out) < 32:
        out.append(cur)
        cur = cur.__cause__
    return out


def is_transient_run_stream_failure(exc: BaseException) -> bool:
    if os.environ.get("TE_MG2_NO_RETRY_ON_403", "").lower() in ("1", "true", "yes"):
        blob403 = " ".join(repr(x) for x in _exception_causes(exc)).lower()
        if "permissiondenied" in blob403 or "error code: 403" in blob403 or " 403 " in blob403:
            return False

    exc_types: list[type[BaseException]] = [TimeoutError, asyncio.TimeoutError]
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        exc_types.extend([APIConnectionError, APITimeoutError, RateLimitError])
    except ImportError:
        pass
    try:
        from openai import PermissionDeniedError

        exc_types.append(PermissionDeniedError)
    except ImportError:
        pass
    try:
        import httpx

        for name in ("ConnectError", "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError"):
            t = getattr(httpx, name, None)
            if isinstance(t, type) and issubclass(t, BaseException):
                exc_types.append(t)
    except ImportError:
        pass

    transient_types = tuple(exc_types)
    for e in _exception_causes(exc):
        if isinstance(e, transient_types):
            return True

    blob = " ".join(repr(x) for x in _exception_causes(exc)).lower()
    for needle in (
        "apiconnectionerror",
        "connection error",
        "connecterror",
        "all connection attempts failed",
        "readtimeout",
        "timeout",
        "ratelimiterror",
        "remoteprotocolerror",
        "permissiondeniederror",
        "error code: 403",
    ):
        if needle in blob:
            return True
    if (
        ("parse_chat_completion" in blob or "_parse_chat_completion" in blob)
        and "nonetype" in blob
        and "iterable" in blob
    ):
        return True
    return False


async def close_surfer_silently(surfer: Any) -> None:
    try:
        await surfer.close()
    except Exception:
        pass


def build_agent_topology(history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    agents_involved: set[str] = set()
    message_flow: Dict[str, List[str]] = {}
    orchestrator_name: str | None = None

    for entry in history:
        name = entry.get("name", "")
        content = entry.get("content", "")

        if name and name not in ("user", "unknown"):
            agents_involved.add(name)

        if "orchestrator" in str(name).lower() or "magenticone" in str(name).lower():
            orchestrator_name = name

        if "sends to" in content:
            parts = content.split("sends to")
            if len(parts) >= 2:
                sender_part = parts[0].strip()
                receiver_part = parts[1].strip()

                sender_words = sender_part.split()
                candidate_sender = sender_words[-1] if sender_words else name

                if name and str(name).lower().endswith(str(candidate_sender).lower()):
                    sender = name
                else:
                    sender = candidate_sender

                if ":" in receiver_part:
                    receiver_full = receiver_part.split(":")[0].strip()
                    receiver = receiver_full.split()[0] if receiver_full else "unknown"
                else:
                    receiver = receiver_part.split()[0] if receiver_part else "unknown"

                receiver = receiver.replace("<", "").replace(">", "")

                message_flow.setdefault(sender, [])
                if receiver not in message_flow[sender] and receiver != "all":
                    message_flow[sender].append(receiver)

    if not message_flow:
        agents_list = sorted(list(agents_involved))
        if agents_list:
            center_node = orchestrator_name if orchestrator_name else "MagenticOneOrchestrator"
            topology: Dict[str, List[str]] = {center_node: agents_list}
        else:
            topology = {"unknown": []}
    else:
        topology = message_flow

    return topology


def apply_built_topology_to_monitor(monitor: Any, topology: Dict[str, List[str]]) -> None:
    """Write edges from :func:`build_agent_topology` into ``BaseMonitor.topology`` via ``record_topology``."""
    if monitor is None:
        return
    record = getattr(monitor, "record_topology", None)
    if not callable(record):
        return
    for src, dsts in topology.items():
        if not isinstance(dsts, list):
            continue
        for dst in dsts:
            s, d = str(src).strip(), str(dst).strip()
            if not s or not d or s == d:
                continue
            record(s, d)


def build_system_prompt(
    topology: Dict[str, List[str]],
    history: List[Dict[str, Any]],
    surfer: Any,
    file_surfer: Any,
    coder: Any,
    terminal: Any,
) -> Dict[str, Any]:
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_ext.agents.file_surfer._file_surfer import FileSurfer as _FileSurferClass
    from autogen_ext.agents.web_surfer._prompts import (
        WEB_SURFER_QA_SYSTEM_MESSAGE,
        WEB_SURFER_TOOL_PROMPT_MM,
        WEB_SURFER_TOOL_PROMPT_TEXT,
    )
    from autogen_ext.agents.magentic_one._magentic_one_coder_agent import (
        MAGENTIC_ONE_CODER_SYSTEM_MESSAGE,
    )

    try:
        from autogen_agentchat.teams._group_chat._magentic_one._prompts import (
            ORCHESTRATOR_SYSTEM_MESSAGE,
            ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
            ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
            ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
            ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
            ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
            ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
            ORCHESTRATOR_FINAL_ANSWER_PROMPT,
        )
    except ImportError:
        ORCHESTRATOR_SYSTEM_MESSAGE = ""
        ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT = ""
        ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT = ""
        ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT = ""
        ORCHESTRATOR_PROGRESS_LEDGER_PROMPT = ""
        ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT = ""
        ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT = ""
        ORCHESTRATOR_FINAL_ANSWER_PROMPT = ""

    enhanced_coder = WINDOWS_CODER_CODE_EXECUTOR_INSTRUCTION + "\n\n" + MAGENTIC_ONE_CODER_SYSTEM_MESSAGE

    system_prompt: Dict[str, Any] = {}
    agents_in_topology: set[str] = set()
    for receivers in topology.values():
        agents_in_topology.update(receivers)
    agents_in_topology.update(topology.keys())

    if any("WebSurfer" in agent for agent in agents_in_topology) and surfer is not None:
        system_prompt["WebSurfer"] = {
            "system_message": WEB_SURFER_QA_SYSTEM_MESSAGE,
            "tool_prompt_template_multimodal": WEB_SURFER_TOOL_PROMPT_MM,
            "tool_prompt_template_text": WEB_SURFER_TOOL_PROMPT_TEXT,
            "description": getattr(surfer, "description", ""),
            "note": "Template; runtime prompt is built in _generate_reply() with page state.",
        }

    if any("FileSurfer" in agent for agent in agents_in_topology):
        sm = ""
        if hasattr(_FileSurferClass, "DEFAULT_SYSTEM_MESSAGES"):
            msgs = _FileSurferClass.DEFAULT_SYSTEM_MESSAGES
            if msgs:
                sm = getattr(msgs[0], "content", "") or ""
        system_prompt["FileSurfer"] = {
            "system_message": sm,
            "description": getattr(file_surfer, "description", ""),
        }

    if any("Coder" in agent or "MagenticOneCoder" in agent for agent in agents_in_topology):
        system_prompt["Coder"] = {
            "system_message": enhanced_coder,
            "description": getattr(coder, "description", ""),
        }

    if any("orchestrator" in agent.lower() or "magenticone" in agent.lower() for agent in agents_in_topology):
        system_prompt["MagenticOneOrchestrator"] = {
            "system_message": ORCHESTRATOR_SYSTEM_MESSAGE,
            "task_ledger_facts_prompt": ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
            "task_ledger_plan_prompt": ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
            "task_ledger_full_prompt": ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
            "progress_ledger_prompt": ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
            "task_ledger_facts_update_prompt": ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
            "task_ledger_plan_update_prompt": ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
            "final_answer_prompt": ORCHESTRATOR_FINAL_ANSWER_PROMPT,
            "description": "Ledger-based MagenticOne orchestrator.",
        }

    if any("ComputerTerminal" in agent or "CodeExecutor" in agent for agent in agents_in_topology):
        system_prompt["ComputerTerminal"] = {
            "system_message": CodeExecutorAgent.DEFAULT_SYSTEM_MESSAGE,
            "description": getattr(terminal, "description", ""),
        }

    for agent_name in agents_in_topology:
        if agent_name not in system_prompt and agent_name != "MagenticOneOrchestrator":
            for entry in history:
                if entry.get("name") == agent_name:
                    system_prompt[agent_name] = {
                        "description": "Agent involved in the execution",
                        "note": "Placeholder (no static template collected).",
                    }
                    break

    return system_prompt


def system_prompt_map_to_strings(prompt_map: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in prompt_map.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def create_magentic_team(
    model_client: Any,
    workspace: Path,
    *,
    include_web: bool = True,
) -> tuple[Any, Any, Any, Any, Any]:
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_agentchat.teams import MagenticOneGroupChat
    from autogen_ext.agents.file_surfer import FileSurfer
    from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
    from autogen_ext.agents.magentic_one._magentic_one_coder_agent import (
        MAGENTIC_ONE_CODER_SYSTEM_MESSAGE,
    )

    enhanced = WINDOWS_CODER_CODE_EXECUTOR_INSTRUCTION + "\n\n" + MAGENTIC_ONE_CODER_SYSTEM_MESSAGE
    surfer: Any = None
    agents: list[Any] = []
    if include_web:
        try:
            from autogen_ext.agents.web_surfer import MultimodalWebSurfer

            surfer = MultimodalWebSurfer("WebSurfer", model_client=model_client)
            agents.append(surfer)
        except ImportError:
            surfer = None
    file_surfer = FileSurfer("FileSurfer", model_client=model_client)
    coder = MagenticOneCoderAgent("Coder", model_client=model_client, system_message=enhanced)
    terminal = CodeExecutorAgent(
        "ComputerTerminal",
        code_executor=make_local_command_line_code_executor(workspace),
    )
    agents.extend([file_surfer, coder, terminal])
    team = MagenticOneGroupChat(agents, model_client=model_client)
    return surfer, file_surfer, coder, terminal, team


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _load_config_from_ini():
    """Load ``config.ini`` and populate environment variables."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "config.ini"

    if not config_path.exists():
        print(f"[warn] Config file not found: {config_path}")
        return

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    if 'magentic' not in config:
        print(f"[warn] Config file missing [magentic] section")
        return

    magentic = config['magentic']

    # API settings
    if magentic.get('api_key'):
        os.environ.setdefault('MAGENTIC_API_KEY', magentic['api_key'])

    if magentic.get('base_url'):
        os.environ.setdefault('MAGENTIC_BASE_URL', magentic['base_url'])

    if magentic.get('model'):
        os.environ.setdefault('MAGENTIC_MODEL', magentic['model'])

    if magentic.get('timeout_sec'):
        os.environ.setdefault('MAGENTIC_TIMEOUT_SEC', magentic['timeout_sec'])

    if magentic.get('max_retries'):
        os.environ.setdefault('MAGENTIC_MAX_RETRIES', magentic['max_retries'])

    # model_info JSON
    if magentic.get('model_info'):
        os.environ.setdefault('MAGENTIC_MODEL_INFO_JSON', magentic['model_info'])

    # GAIA dataset paths
    if magentic.get('gaia_validation_dir'):
        os.environ.setdefault('GAIA_VALIDATION_DIR', magentic['gaia_validation_dir'])

    if magentic.get('gaia_attachments_root'):
        os.environ.setdefault('GAIA_ATTACHMENTS_ROOT', magentic['gaia_attachments_root'])

    # Run / retry tuning
    if magentic.get('run_max_attempts'):
        os.environ.setdefault('TE_MG2_RUN_MAX_ATTEMPTS', magentic['run_max_attempts'])

    if magentic.get('run_retry_base_sec'):
        os.environ.setdefault('TE_MG2_RUN_RETRY_BASE_SEC', magentic['run_retry_base_sec'])

    print(f"[config] Loaded settings from {config_path}")