"""Codex（OpenAI Codex CLI）runtime adapter。

非交互回合走 `codex exec --json`：事件是逐行 JSONL（现场采样 0.155.1 钉死），
`thread.started.thread_id` 是 Codex 侧会话 id，`item.completed` 的 `agent_message` 是正文，
`command_execution` 是工具执行。resume 用 `codex exec resume <thread_id>`。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .base import (
    ActionResult, Diagnostic, QuestionAnswer, RunRequest, RunResult, RuntimeDescriptor,
    RuntimeEvent, RuntimeHealth, ServiceAction, SetupContext, SetupResult,
)
from .common import PROJECT_ROOT, runtime_env

# 非交互沙箱与审批（现场采样确认可用、不挂起）：
# - workspace-write 允许写工作区；network_access 让技能脚本能联网（Codex 在 workspace-write
#   下默认禁网，而 Easel 的抓取/发布/生图技能都要出网）。
# - approval_policy=never 保证无 TTY 时不卡审批；只作用于子进程，不改用户全局配置。
# - resume 子命令没有 -s/--cd，只能用 -c 覆盖 sandbox_mode。
_SANDBOX = ["-s", "workspace-write"]
_SANDBOX_RESUME = ["-c", 'sandbox_mode="workspace-write"']
_RUNTIME_OVERRIDES = ["-c", 'approval_policy="never"',
                      "-c", "sandbox_workspace_write.network_access=true"]
_TOOL_ITEM_TYPES = {"command_execution", "mcp_tool_call", "web_search", "file_change"}


def base_cmd() -> list[str]:
    executable = shutil.which("codex")
    if not executable:
        raise FileNotFoundError("Codex CLI 未找到；请安装：npm install -g @openai/codex@latest")
    if Path(executable).suffix.lower() in (".cmd", ".bat"):
        # npm 包 @openai/codex 的 bin 是 bin/codex.js：Windows 下绕开 .cmd shim，交给 node。
        js = Path(executable).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        if js.is_file():
            return ["node", str(js)]
    return [executable]


def session_path(session_key: str, sessions_dir: Path) -> Path:
    return sessions_dir / f"codex-{hashlib.sha256(session_key.encode()).hexdigest()[:24]}.txt"


def codex_version() -> str:
    """`codex --version` 的版本号（如 0.155.1）；失败返回空。"""
    try:
        result = subprocess.run(base_cmd() + ["--version"], capture_output=True, text=True,
                                timeout=10, cwd=PROJECT_ROOT, env=runtime_env())
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    line = (result.stdout or "").strip().splitlines()
    return line[0].removeprefix("codex-cli ").strip()[:40] if line else ""


def session_id(session_key: str, sessions_dir: Path | None) -> str | None:
    if sessions_dir is None:
        return None
    try:
        value = session_path(session_key, sessions_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _error_message(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error).strip()
    return str(error or event.get("message") or "Codex 回合失败").strip()


def event_from(data: dict) -> RuntimeEvent | None:
    """把一条已解析的 JSONL 事件映射为 RuntimeEvent；错误事件抛 RuntimeError。"""
    kind = str(data.get("type") or "")
    if kind in ("error", "turn.failed"):
        raise RuntimeError(_error_message(data) or "Codex 回合失败")
    if kind not in ("item.started", "item.completed"):
        return None
    item = data.get("item") if isinstance(data.get("item"), dict) else {}
    item_type = str(item.get("type") or "")
    text = str(item.get("text") or "")
    if item_type == "agent_message" and text:
        return RuntimeEvent("text", text)
    if item_type == "reasoning" and text:
        return RuntimeEvent("thinking", text)
    if kind == "item.started" and item_type in _TOOL_ITEM_TYPES:
        return RuntimeEvent("activity", "🔧 正在执行命令…" if item_type == "command_execution"
                            else "🔧 正在调用工具…")
    return None


def parse_event(line: str) -> RuntimeEvent | None:
    try:
        data = json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return event_from(data)


class CodexRunHandle:
    def __init__(self, process: subprocess.Popen, sessions_path: Path | None):
        self.process = process
        self.sessions_path = sessions_path
        self.session_id: str | None = None
        self._text: list[str] = []
        self._usage: dict = {}
        self._noise: list[str] = []
        self._error: str | None = None
        self._events_done = False

    def _remember_session(self, thread_id: str) -> None:
        self.session_id = thread_id
        if self.sessions_path is None:
            return
        try:
            self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
            self.sessions_path.write_text(thread_id, encoding="utf-8")
        except OSError:
            pass

    def _note_noise(self, line: str) -> None:
        self._noise.append(line)
        del self._noise[:-20]

    def events(self):
        if self._events_done:
            return
        assert self.process.stdout is not None
        for line in self.process.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                self._note_noise(raw)
                continue
            if not isinstance(data, dict):
                self._note_noise(raw)
                continue
            kind = data.get("type")
            if kind == "thread.started":
                thread_id = str(data.get("thread_id") or "").strip()
                if thread_id:
                    self._remember_session(thread_id)
                continue
            if kind == "turn.completed":
                usage = data.get("usage")
                if isinstance(usage, dict):
                    self._usage = usage
                continue
            try:
                event = event_from(data)
            except RuntimeError as exc:
                self._error = str(exc)
                continue
            if event is None:
                continue
            if event.type == "text":
                self._text.append(event.text)
            yield event
        self._events_done = True

    def wait(self) -> RunResult:
        if not self._events_done:
            for _ in self.events():
                pass
        rc = self.process.wait()
        diagnostics = {"session_id": self.session_id, "usage": self._usage,
                       "noise_tail": self._noise[-1][:300] if self._noise else ""}
        if self._error or (rc == 0 and not self._text):
            message = self._error or diagnostics["noise_tail"] or \
                "Codex 已退出，但没有返回正文；请检查登录状态或重试。"
            diagnostics["error"] = message
            return RunResult(rc or 1, "".join(self._text), clean_end=False,
                             stop_reason="runtime_error", diagnostics=diagnostics)
        if rc != 0:
            diagnostics["error"] = diagnostics["noise_tail"] or f"Codex 退出码 {rc}"
            return RunResult(rc, "".join(self._text), clean_end=False,
                             stop_reason="runtime_error", diagnostics=diagnostics)
        return RunResult(rc, "".join(self._text), clean_end=True, diagnostics=diagnostics)

    def cancel(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def poll(self) -> int | None:
        return self.process.poll()


class CodexAdapter:
    descriptor = RuntimeDescriptor(
        "codex", "Codex", "npm install -g @openai/codex@latest",
        frozenset({"interactive", "session_delete", "native_abort", "thinking_events",
                   "activity_events", "native_stream"}),
    )

    def setup(self, context: SetupContext) -> SetupResult:
        if shutil.which("codex") is None:
            result = subprocess.run(["npm", "install", "-g", "@openai/codex@latest", "--loglevel", "warn"],
                                    cwd=context.project_root, env=context.env or None)
            if result.returncode:
                return SetupResult(False, "Codex 安装失败")
        try:
            base_cmd()
        except FileNotFoundError as exc:
            return SetupResult(False, str(exc))
        return SetupResult(True, "Codex 已就绪（登录与模型复用本机 codex 配置）")

    def open_chat(self, prompt: str | None) -> int:
        cmd = base_cmd()
        if prompt:
            cmd.append(prompt)
        return subprocess.run(cmd, cwd=PROJECT_ROOT, env=runtime_env()).returncode

    def start(self, run: RunRequest) -> CodexRunHandle:
        # 模型与思考强度来自项目 .env：会话会记住旧设置，每轮显式传 -m / -c 才能让面板
        # 保存的改动下一条消息生效（延迟导入避免模块循环）。
        from .codex_config import model as easel_model, reasoning as easel_reasoning
        cmd = base_cmd() + ["exec"]
        sessions_path = session_path(run.session_key, run.sessions_dir) if run.sessions_dir else None
        sid = session_id(run.session_key, run.sessions_dir)
        if sid:
            cmd += ["resume", sid, *_SANDBOX_RESUME]
        else:
            cmd += ["-C", str(run.cwd), *_SANDBOX]
        cmd += ["--json", "--skip-git-repo-check", *_RUNTIME_OVERRIDES]
        model = easel_model()
        if model:
            cmd += ["-m", model]
        reasoning = easel_reasoning()
        if reasoning:
            cmd += ["-c", f'model_reasoning_effort="{reasoning}"']
        cmd.append(run.prompt)
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=run.cwd, text=True, bufsize=1,
                                env=run.env)
        return CodexRunHandle(proc, sessions_path)

    def health(self) -> RuntimeHealth:
        from .codex_config import login_status
        if shutil.which("codex") is None:
            return RuntimeHealth(False, self.descriptor.install_hint)
        logged_in, detail = login_status()
        return RuntimeHealth(logged_in, detail if logged_in else detail or "未登录；请在终端运行 codex login")

    def diagnose(self) -> list[Diagnostic]:
        from .codex_config import auth_state, config_model, doctor_report, login_status, model as easel_model
        command_ok = shutil.which("codex") is not None
        version = codex_version() if command_ok else ""
        report = doctor_report() if command_ok else {}
        if report:
            logged_in, auth_mode = auth_state(report)
        elif command_ok:
            # doctor 不可用时降级到 codex login status，避免把已登录误报成未登录
            logged_in, auth_mode = login_status()
        else:
            logged_in, auth_mode = False, ""
        model = easel_model() or config_model(report)
        skills_ok = (PROJECT_ROOT / "skills" / "openclaw").is_dir()
        return [
            Diagnostic(f"Codex command ({version})" if version else "Codex command",
                       command_ok, self.descriptor.install_hint),
            Diagnostic(f"Codex login ({auth_mode})" if auth_mode else "Codex login",
                       logged_in, "在终端运行 codex login"),
            Diagnostic(f"Codex model ({model})" if model else "Codex model", bool(model),
                       "在 Web 设置面板选择默认模型，或运行 codex 配置本机模型"),
            Diagnostic("Skills synced", skills_ok, "重新运行 setup"),
        ]

    def node_requirement(self) -> tuple[bool, str]:
        # npm 包 @openai/codex 的 engines：node >=16（brew 装的单文件版不需要 Node）。
        return False, "16"

    def is_local_gateway_base(self, url: str) -> bool:
        return False

    def provider_creds(self) -> dict[str, tuple[str, str]]:
        return {}

    def sync_chat_providers(self, provider_updates: dict[str, dict], keep_custom: set[str],
                            primary_ref: str) -> str:
        return "当前 runtime（Codex）不支持同步 chat 供应商"

    def config_snapshot(self) -> dict:
        return {"primary": "", "providers": {}}

    def manage_service(self, action: ServiceAction) -> ActionResult:
        return ActionResult.unsupported("gateway 常驻服务（Codex 每轮直接运行 codex exec）")

    def delete_session(self, session_key: str, sessions_dir: Path | None = None) -> ActionResult:
        if sessions_dir is None:
            return ActionResult(False, "sessions directory is required")
        sid = session_id(session_key, sessions_dir)
        if not sid:
            return ActionResult(False, "session not found", data={"deleted": False})
        try:
            result = subprocess.run(base_cmd() + ["delete", sid], capture_output=True, text=True,
                                    timeout=30, cwd=PROJECT_ROOT, env=runtime_env())
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ActionResult(False, str(exc), data={"deleted": False})
        session_path(session_key, sessions_dir).unlink(missing_ok=True)
        if result.returncode != 0:
            return ActionResult(False, f"codex delete 退出码 {result.returncode}",
                                data={"deleted": True})
        return ActionResult(True, data={"deleted": True})

    def answer_question(self, request: QuestionAnswer) -> ActionResult:
        return ActionResult.unsupported("questions")

    def question_status(self, question_ids: list[str]) -> ActionResult:
        return ActionResult.unsupported("questions")
