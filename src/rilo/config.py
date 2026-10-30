from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


REMOTE_ROBOT_BACKEND = "remote_robot"
REMOTE_ROBOT_CONNECTION_ONLY_REASON = (
    "원격 로봇은 로봇 client / 정책 서버 패널에서 SSH 연결하고 실행하세요. 로컬 Runner·자동 파이프라인은 사용할 수 없습니다."
)


@dataclass(frozen=True)
class Settings:
    backend: str = "libero"
    policy: str = "openai"
    model: str = "gpt-6-astra"
    api_key: str = field(default="", repr=False)
    base_url: str | None = field(default=None, repr=False)
    reasoning_effort: str = "medium"
    api_timeout: float = 240.0
    max_output_tokens: int = 16384
    prompt_file: Path | None = None
    suite: str = "libero_goal"
    task_id: int = 0
    image_size: int = 512
    data_dir: Path = Path("artifacts/runs")
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            backend=os.getenv("RILO_ENV", "libero"),
            policy=os.getenv("RILO_POLICY", "openai"),
            model=os.getenv("OPENAI_MODEL", "gpt-6-astra"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            api_timeout=float(os.getenv("RILO_API_TIMEOUT", "240")),
            max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "16384")),
            prompt_file=Path(os.environ["RILO_PROMPT_FILE"]) if os.getenv("RILO_PROMPT_FILE") else None,
            suite=os.getenv("LIBERO_SUITE", "libero_goal"),
            task_id=int(os.getenv("RILO_TASK_ID", "0")),
            image_size=int(os.getenv("RILO_IMAGE_SIZE", "512")),
            data_dir=Path(os.getenv("RILO_DATA_DIR", "artifacts/runs")),
            host=os.getenv("RILO_HOST", "127.0.0.1"),
            port=int(os.getenv("RILO_PORT", "8000")),
        )


def make_adapter(settings: Settings):
    if settings.backend == "libero":
        from rilo.adapters.libero import LiberoAdapter
        return LiberoAdapter(suite=settings.suite, image_size=settings.image_size)
    if settings.backend == "demo":
        from rilo.adapters.demo import DemoAdapter
        return DemoAdapter(image_size=settings.image_size)
    if ":" in settings.backend:
        # Server-side configuration only. The browser cannot choose import paths.
        from importlib import import_module
        module, name = settings.backend.split(":", 1)
        return getattr(import_module(module), name)()
    raise ValueError("RILO_ENV must be libero, demo, or module:adapter_factory")


def make_policy(settings: Settings):
    if settings.policy == "openai":
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it in the server environment and restart RILO.")
        from rilo.policies.openai_policy import OpenAIPolicy
        return OpenAIPolicy(model=settings.model, api_key=settings.api_key,
                            base_url=settings.base_url, reasoning_effort=settings.reasoning_effort,
                            timeout=settings.api_timeout, max_output_tokens=settings.max_output_tokens,
                            prompt_file=settings.prompt_file)
    if settings.policy == "scripted":
        if settings.backend != "demo":
            raise ValueError("The scripted smoke-test policy is only available with RILO_ENV=demo")
        from rilo.policies.scripted import ScriptedPolicy
        return ScriptedPolicy()
    if ":" in settings.policy:
        from importlib import import_module
        module, name = settings.policy.split(":", 1)
        return getattr(import_module(module), name)()
    raise ValueError("RILO_POLICY must be openai, scripted, or module:policy_factory")
