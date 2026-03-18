from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def get_repo_root(start: Path | None = None) -> Path:
    if start is None:
        return Path(__file__).resolve().parents[3]
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / 'run.py').exists() and (candidate / 'src' / 'quant_lab').exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def get_project_env_path(repo_root: Path | None = None) -> Path:
    return get_repo_root(repo_root) / '.env'


def load_project_env(repo_root: Path | None = None, override: bool = False) -> Path | None:
    env_path = get_project_env_path(repo_root)
    if not env_path.exists():
        return None
    load_dotenv(dotenv_path=env_path, override=override)
    return env_path


def get_required_env(name: str, repo_root: Path | None = None) -> str:
    load_project_env(repo_root=repo_root, override=False)
    value = str(os.getenv(name, '')).strip()
    if value:
        return value
    env_path = get_project_env_path(repo_root)
    raise ValueError(f"{name} not found. Set it in the repo-root .env at {env_path} or export it in the shell.")
